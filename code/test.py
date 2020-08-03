import argparse
import numpy as np
import os
import time
from pprint import pprint

import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from data import create_dataloader
from models import deepdeblur
from utils.utils import save_image, AverageMeter


parser = argparse.ArgumentParser()

# settings
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu id to use')
parser.add_argument('--n_cpu', type=int, default=4,
                    help='cpus for data processing')

# data parameters
parser.add_argument('--dataset', type=str, default='GOPRO_Large',
                    help='GOPRO | GOPRO_Large')
parser.add_argument('--blur_type', type=str, default='gamma',
                    help='gamma | lin')

parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='path to the checkpoint to load')
parser.add_argument('--save_dir', type=str, default=None,
                    help='directory name to save results (default: YY-MM-DD-HHMMSS)')

def main():
    args = parser.parse_args()
    pprint(vars(args))

    args.dataset = os.path.join('./datasets', args.dataset)

    if args.save_dir is None:
        args.save_dir = time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))
    args.save_dir = os.path.join('./results', args.save_dir)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.makedirs('results', exist_ok=True)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        raise ValueError('save dir exists!')

    test_loader = create_dataloader(args, phase='test')

    model = deepdeblur.DeepDeblur_scale3()
    model.to(device)

    if args.checkpoint_path is None:
        print('no checkpoint is specified.. load pre-trained model')
        args.checkpoint_path = './checkpoints/pretrained/430.pth'
    pretrained_dict = torch.load(args.checkpoint_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

    test(test_loader, model, device, args)

def test(test_loader, model, device, args):
    test_psnr = AverageMeter('PSNR')
    test_ssim = AverageMeter('SSIM')

    model.eval()
    with torch.no_grad():
        for i, (input, label) in enumerate(test_loader, 1):
            blur1, blur2, blur3 = input
            sharp1, sharp2, sharp3 = label
            blur1, blur2, blur3 = blur1.to(device), blur2.to(device), blur3.to(device)
            sharp1, sharp2, sharp3 = sharp1.to(device), sharp2.to(device), sharp3.to(device)

            pred1, _, _ = model(blur1, blur2, blur3)

            sharp1 = sharp1.detach().clone().cpu().numpy().squeeze().transpose(1, 2, 0)
            pred1 = pred1.detach().clone().cpu().numpy().squeeze().transpose(1, 2, 0)
            blur1 = blur1.detach().clone().cpu().numpy().squeeze().transpose(1, 2, 0)
            
            sharp1 += 0.5
            pred1 += 0.5
            blur1 += 0.5

            psnr = peak_signal_noise_ratio(
                sharp1, pred1, 
                data_range=1.
            )
            ssim = structural_similarity(
                sharp1, pred1, 
                multichannel=True, 
                gaussian_weights=True, 
                use_sample_covariance=False,
                data_range=1.
            )

            test_psnr.update(psnr)
            test_ssim.update(ssim)

            print('{:d}/{:d} | PSNR (dB) {:.2f} | SSIM {:.4f}'.format(i, len(test_loader), psnr, ssim))
            
            save_image(sharp1, os.path.join(args.save_dir, '{:d}_sharp.png'.format(i)))
            save_image(blur1, os.path.join(args.save_dir, '{:d}_blur.png'.format(i)))
            save_image(pred1, os.path.join(args.save_dir, '{:d}_pred.png'.format(i)))

    print('>> Avg PSNR, SSIM: {:.2f}, {:.2f}'.format(test_psnr.avg, test_ssim.avg))

        
if __name__ == '__main__':
    main()