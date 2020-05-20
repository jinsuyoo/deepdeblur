import argparse
import os
import random
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

from data import create_dataloader
from models import deepdeblur, discriminator
from utils.utils import get_summarywriter, print_args, AverageMeter


parser = argparse.ArgumentParser()

# settings
parser.add_argument('--seed', type=int, default=None,
                    help='seed for initializing training')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu id to use')
parser.add_argument('--n_cpu', type=int, default=8,
                    help='cpus for data processing')
parser.add_argument('--experiment_id', type=str, default=None,
                    help='checkpoint directory name (default: YY-MM-DD-HHMMSS)')

# training arguments
parser.add_argument('--n_epochs', type=int, default=450,
                    help='number of total epochs')
parser.add_argument('--n_iter_per_epoch', type=int, default=1000,
                    help='number of iterations per single epoch')
parser.add_argument('--batch_size', type=int, default=4,
                    help='mini-batch size')
parser.add_argument('--lr', type=float, default=5e-5,
                    help='initial learning rate')
parser.add_argument('--lr_decay_step', type=int, default=150,
                    help='learning rate decay step')
parser.add_argument('--adv_lambda', type=float, default=1e-4,
                    help='adversarial loss weight constant')

# data arguments
parser.add_argument('--dataset', type=str, default='GOPRO_Large',
                    help='GOPRO | GOPRO_Large')
parser.add_argument('--blur_type', type=str, default='gamma',
                    help='gamma | lin')
parser.add_argument('--crop_size', type=int, default=256,
                    help='crop size')

# others
parser.add_argument('--log_interval', type=int, default=100,
                    help='log interval, iteration (default: 100)')
parser.add_argument('--val_interval', type=int, default=10,
                    help='validate interval, epoch (default: 10)')
parser.add_argument('--tensorboard', action='store_true',
                    help='visualize training?')


def main():
    args = parser.parse_args()

    print_args(args)

    if args.seed is not None:
        print('seed number given => {:d}'.format(args.seed))
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    args.dataset = os.path.join('./datasets', args.dataset)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.experiment_id is not None:
        model_id = args.experiment_id
    else:
        model_id = time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))
    args.save_path = os.path.join('checkpoints', model_id)
    os.makedirs(args.save_path, exist_ok=True)
    print('experiment id => {} \ncheckpoint path => {}'.format(model_id, args.save_path))

    writer = get_summarywriter(model_id) if args.tensorboard else None

    inf_train_loader = create_dataloader(args, phase='train', inf=True)
    val_loader = create_dataloader(args, phase='test')

    model = deepdeblur.DeepDeblur_scale3()
    model.to(device)
    netD = discriminator.Discriminator()
    netD.to(device)

    train(inf_train_loader, val_loader, model, netD, device, writer, args)
    
    writer.close()


def validate(current_epoch, data_loader, model, device, writer, args):
    val_psnr = AverageMeter('PSNR')

    model.eval()
    with torch.no_grad():
        for i, (input, label) in enumerate(tqdm(data_loader)):
            blur1, blur2, blur3 = input
            blur1, blur2, blur3 = blur1.to(device), blur2.to(device), blur3.to(device)
            sharp1, _, _ = label
            sharp1 = sharp1.to(device)

            pred1, _, _ = model(blur1, blur2, blur3)

            sharp1 = sharp1.detach().clone().cpu().numpy().squeeze().transpose(1, 2, 0)
            pred1 = pred1.detach().clone().cpu().numpy().squeeze().transpose(1, 2, 0)

            np.clip(pred1, 0, 1.)

            sharp1 += 0.5
            pred1 += 0.5

            psnr = peak_signal_noise_ratio(sharp1, pred1, data_range=1.)

            val_psnr.update(psnr)
            
    print('>> PSNR on validate dataset: ', val_psnr.avg)
    if writer is not None:
        writer.add_scalar('Validate/'+val_psnr.name, val_psnr.avg, current_epoch)


def train(inf_train_loader, val_loader, model, netD, device, writer, args):
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    g_optimizer = optim.Adam(model.parameters(), args.lr, 
                             betas=(0.9,0.999), 
                             eps=1e-8, 
                             weight_decay=0)

    d_optimizer = optim.Adam(netD.parameters(), args.lr, 
                             betas=(0.9,0.999), 
                             eps=1e-8, 
                             weight_decay=0)

    g_scheduler = optim.lr_scheduler.StepLR(
        g_optimizer, 
        step_size=args.lr_decay_step,
        gamma=0.1)
    
    d_scheduler = optim.lr_scheduler.StepLR(
        d_optimizer, 
        step_size=args.lr_decay_step,
        gamma=0.1)
    
    netD.train()

    real_label = torch.ones([args.batch_size, 1], device=device)
    fake_label = torch.zeros([args.batch_size, 1], device=device)

    d_losses = AverageMeter('D loss')
    total_losses = AverageMeter('Total G loss')
    content_losses = AverageMeter('Content loss')
    adv_losses = AverageMeter('Adversarial loss')

    print('start training..')
    for epoch in range(1, args.n_epochs + 1):
        model.train()

        for i in range(1, args.n_iter_per_epoch + 1):
            ########################
            # Update Discriminator #
            ########################
            for p in netD.parameters():
                p.requires_grad = True

            inputs, labels = next(inf_train_loader)

            blur1, blur2, blur3 = inputs
            blur1, blur2, blur3 = blur1.to(device), blur2.to(device), blur3.to(device)
            sharp1, _, _ = labels
            sharp1 = sharp1.to(device)

            pred1, _, _ = model(blur1, blur2, blur3)

            # compute adversarial loss
            real_loss = bce_loss(netD(sharp1), real_label)
            fake_loss = bce_loss(netD(pred1.detach()), fake_label)
            d_loss = real_loss + fake_loss

            # update discriminator
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            ###########################
            # Update DeepDeblur model #
            ###########################
            for p in netD.parameters():
                p.requires_grad = False

            inputs, labels = next(inf_train_loader)

            blur1, blur2, blur3 = inputs
            blur1, blur2, blur3 = blur1.to(device), blur2.to(device), blur3.to(device)
            sharp1, sharp2, sharp3 = labels
            sharp1, sharp2, sharp3 = sharp1.to(device), sharp2.to(device), sharp3.to(device)

            pred1, pred2, pred3 = model(blur1, blur2, blur3)

            # compute content loss
            mse_loss_1 = mse_loss(pred1, sharp1)
            mse_loss_2 = mse_loss(pred2, sharp2) 
            mse_loss_3 = mse_loss(pred3, sharp3)

            # compute total losses for generator
            content_loss = (mse_loss_1 + mse_loss_2 + mse_loss_3) / 3
            adv_loss = bce_loss(netD(pred1), real_label)

            total_g_loss = content_loss + args.adv_lambda * adv_loss

            # update generator
            g_optimizer.zero_grad()
            total_g_loss.backward()
            g_optimizer.step()

            ###########
            # Logging #
            ###########           
            d_losses.update(d_loss.item())
            total_losses.update(total_g_loss.item())
            content_losses.update(content_loss.item())
            adv_losses.update(adv_loss.item())

            if i % args.log_interval == 0:
                print('Epoch {:d}/{:d} | Iteration {:d}/{:d} | D loss {:.6f} | Total G loss {:.6f} | Content loss {:.6f} | Adversarial loss {:.6f}'.format(
                    epoch, args.n_epochs, i, args.n_iter_per_epoch, d_losses.avg, total_losses.avg, content_losses.avg, adv_losses.avg
                ))
                if writer is not None:
                    writer.add_scalar('Train/'+d_losses.name, d_losses.avg, d_losses.count)
                    writer.add_scalar('Train/'+total_losses.name, total_losses.avg, total_losses.count)
                    writer.add_scalar('Train/'+content_losses.name, content_losses.avg, content_losses.count)
                    writer.add_scalar('Train/'+adv_losses.name, adv_losses.avg, adv_losses.count)

        if epoch % args.val_interval == 0:
            validate(epoch, val_loader, model, device, writer, args)
            torch.save(model.state_dict(), os.path.join(args.save_path, '{:d}.pth'.format(epoch)))
        
        g_scheduler.step()
        d_scheduler.step()
        
        
if __name__ == '__main__':
    main()