'''
paper:
X. Chen, Z. Chen, J. Xin, L. Guo, W. Zhai, Discriminative condition-guided generative model for induction motor fault diagnosis with limited data,
IEEE Transactions on Industrial Informatics, 22 (2026) 1359–1370.

code: https://github.com/xinswjtu/DCGM-IFD
'''

import time
import argparse
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import ours_model
from utils.losses import *
from utils.utils import *
from utils.supcontrast_loss import SupContrastLoss


net_G = {'ours_model': ours_model.Generator}
net_D = {'ours_model': ours_model.Discriminator}
loss_fns = {'hinge': HingeLoss}

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # model
    parser.add_argument('--model', type=str, default='ours_model', choices=['ours_model'])
    parser.add_argument('--epochs', type=int, default=10, help='max number of epoch')
    # trade-off coefficient
    parser.add_argument('--cond_lambda', type=float, default=0.5, help='trade-off')
    # contrastive loss coefficient
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--lambda_sc_min', type=float, default=0.0)
    parser.add_argument('--lambda_sc_max', type=float, default=1.0)
    parser.add_argument('--lambda_sc_k', type=float, default=0.001)
    # optim、loss、learning rate
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size of the training process')
    parser.add_argument('--n_critic', type=int, default=3, help='every epoch train D for n_critic')
    parser.add_argument('--lr_g', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_d', type=float, default=4e-4, help='learning rate')
    parser.add_argument('--betas_g', nargs='+', type=float, default=(0.1, 0.999), help='for Adam')
    parser.add_argument('--betas_d', nargs='+', type=float, default=(0.1, 0.999), help='for Adam')
    parser.add_argument('--loss', type=str, default='hinge', choices=['hinge'], help='Gan loss')
    # datasets
    parser.add_argument('--dataset_name', type=str, default='Motor')
    parser.add_argument('--n_class', type=int, default=5, help='num. of categorises')
    parser.add_argument('--n_train', type=int, default=20, help='number of training sets')
    parser.add_argument('--z_dim', type=int, default=100, help='noise dimension')
    parser.add_argument('--n_channel', type=int, default=3, help='channel')
    parser.add_argument('--img_size', type=int, default=64, help=' H x W')
    ## generation fake
    parser.add_argument('--eval_model', type=str, default='gan.pth')
    parser.add_argument('--n_gen', type=int, default=50, help='number of generating fake samples')
    # default
    parser.add_argument('--device', type=torch.device, default=torch.device('cuda'))
    args = parser.parse_args()
    args.real_folder = f'./datasets/{args.dataset_name}/real_images'
    return args


def train(args):
    train_x, train_y, train_ds = get_dataset(args, data_type='train')
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    looper = infiniteloop(train_dl)

    G = net_G[args.model](args.z_dim, args.n_class, args.img_size, args.n_channel).to(args.device)
    D = net_D[args.model](args.n_class, args.img_size, args.n_channel).to(args.device)

    optimizer_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=args.betas_g)
    optimizer_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=args.betas_d)

    loss_fn = loss_fns[args.loss]().to(args.device) # adversarial loss: hinge loss
    cond_loss_fn = CrossEntropyLoss().to(args.device) # auxililary classification loss
    supcontrast_fn = SupContrastLoss(args.temperature, args.device).to(args.device) # supervised contrastive loss

    loss = init_dict('dloss', 'adv_dloss', 'c_dloss', 'c_dacc', 'gloss', 'adv_gloss', 'c_gloss', 'c_gacc', 'sc_gloss', 'c_loss')
    epoch_sc = []
    for epoch in range(args.epochs):
        lambda_sc = lambda_sigmoid(epoch, args.lambda_sc_min, args.lambda_sc_max, args.lambda_sc_k, args.epochs//2)
        step, count = 0, 0

        epoch_dloss, epoch_adv_dloss, epoch_c_dloss, epoch_c_dacc, epoch_sc_gloss = [Metric() for _ in range(5)]
        epoch_gloss, epoch_adv_gloss, epoch_c_gloss, epoch_c_gacc = [Metric() for _ in range(4)]

        while step < max(len(train_dl), args.n_critic):
            """ Update D """
            make_GAN_trainable(G, D)
            toggle_grad(D, True)
            toggle_grad(G, False)
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()

            step += 1
            imgs, labels = next(looper)
            n, imgs, labels = imgs.shape[0], imgs.to(args.device), labels.to(args.device)
            with torch.no_grad():
                z = torch.normal(0, 1, (n, args.z_dim)).to(args.device)
                y = torch.randint(args.n_class, (n,)).to(args.device)
                fake_imgs = G(z, y).detach()

            real_dict = D(imgs, labels)
            fake_dict = D(fake_imgs, y, adc_fake=True)

            d_adv_loss = loss_fn(real_dict['adv_output'], fake_dict['adv_output'])

            real_cond_loss = cond_loss_fn(**real_dict)
            d_aux_loss = args.cond_lambda * real_cond_loss

            fake_cond_loss = cond_loss_fn(**fake_dict)
            d_aux_loss += args.cond_lambda * fake_cond_loss

            d_total_loss = d_adv_loss + d_aux_loss
            d_total_loss.backward()
            optimizer_d.step()

            epoch_c_dloss.update(d_aux_loss.item())
            epoch_adv_dloss.update(d_adv_loss.item())
            epoch_dloss.update(d_total_loss.item())

            if (step % args.n_critic) == 0:
                count += 1
                """ Update G """
                make_GAN_trainable(G, D)
                toggle_grad(D, False)
                toggle_grad(G, True)
                optimizer_g.zero_grad()

                z = torch.normal(0, 1, (n, args.z_dim)).to(args.device)
                y = torch.randint(args.n_class, (n,)).to(args.device)
                fake_imgs = G(z, y)
                fake_dict = D(fake_imgs, y)

                g_adv_loss = loss_fn(fake_dict['adv_output'])

                fake_cond_loss = cond_loss_fn(**fake_dict)
                g_aux_loss = args.cond_lambda * fake_cond_loss

                adc_fake_dict = D(fake_imgs, y, adc_fake=True)
                adc_fake_cond_loss = -cond_loss_fn(**adc_fake_dict)
                g_aux_loss += args.cond_lambda * adc_fake_cond_loss
                g_total_loss = g_adv_loss + g_aux_loss

                epoch_c_gloss.update(g_aux_loss.item())
                epoch_adv_gloss.update(g_adv_loss.item())

                x_new = torch.cat([real_dict['h'].detach(), fake_dict['h']])
                y_new = torch.cat([labels, y])
                supcontrast_loss = supcontrast_fn(x_new, y_new)
                epoch_sc_gloss.update(supcontrast_loss.item())

                g_total_loss += lambda_sc * supcontrast_loss
                g_total_loss.backward()
                optimizer_g.step()

                epoch_gloss.update(g_total_loss.item())


        loss['dloss'].append(epoch_dloss.avg)
        loss['gloss'].append(epoch_gloss.avg)
        loss['adv_dloss'].append(epoch_adv_dloss.avg)
        loss['adv_gloss'].append(epoch_adv_gloss.avg)
        loss['c_dloss'].append(epoch_c_dloss.avg)
        loss['c_gloss'].append(epoch_c_gloss.avg)
        info = f"{epoch}/{args.epochs}, d_loss:{epoch_dloss.avg:.4f}, g_loss:{epoch_gloss.avg:.4f}, adv_dloss:{epoch_adv_dloss.avg:.4f}, " \
               f"adv_gloss:{epoch_adv_gloss.avg:.4f}, c_dloss:{epoch_c_dloss.avg:.4f}, c_gloss:{epoch_c_gloss.avg:.4f}"

        loss['sc_gloss'].append(epoch_sc_gloss.avg)
        info += f', sc_gloss:{epoch_sc_gloss.avg:.4f}'
        print(info)

    # save model parameters
    ckpt = {'G': G.state_dict(), 'D': D.state_dict()}
    torch.save(ckpt, os.path.join(rf'./logs/{args.dataset_name}', f'gan.pth'))
    np.save(os.path.join(rf'./logs/{args.dataset_name}', f'gan_loss.npy'), loss)
    return G


@torch.no_grad()
def evaluate(args, G, fake_folder):
    noise = torch.normal(0, 1, (args.n_gen*args.n_class, args.z_dim)).to(args.device)
    label = torch.tensor([[k]*args.n_gen for k in range(args.n_class)],
                         dtype=torch.long).view(-1).to(args.device)
    fake = G(noise, label)
    tensor2img(fake, label, os.path.join(fake_folder, f'fake_images'), zero_one=False)

def main():
    args = parse_args()
    G = train(args)

    # evaluation
    fake_folder = rf'./logs/{args.dataset_name}'
    evaluate(args, G, fake_folder)


if __name__ == "__main__":
    main()





