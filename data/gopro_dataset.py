import copy
import glob
import numpy as np
import os
import random

import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


def augment(input, label, crop_size, mu=0, sigma=2/255.):
    label_input = np.random.randint(0, 10) == 0
    change_saturation = np.random.randint(0, 10) == 0
    flip_h = np.random.randint(0, 2) == 0
    angle = random.choice([0, 90, 180, 270])
    
    shuffle_color = True
    add_noise = False
    label_input= False
  
    if label_input:
        input = copy.deepcopy(label)
    
    # Random crop
    if crop_size != -1:
        h, w = input.size
        rnd_h = random.randint(0, h - crop_size)
        rnd_w = random.randint(0, w - crop_size)
        input = TF.crop(input, rnd_w, rnd_h, crop_size, crop_size)
        label = TF.crop(label, rnd_w, rnd_h, crop_size, crop_size)

    if flip_h:
        input = TF.hflip(input)
        label = TF.hflip(label)
    
    if angle > 0:
        input = TF.rotate(input, angle)
        label = TF.rotate(label, angle)
    
    if change_saturation:
        saturation_factor = 1 + np.random.uniform(-0.5, 0.5)
        
        input = TF.adjust_saturation(input, saturation_factor)
        label = TF.adjust_saturation(label, saturation_factor)
    
    # Augmentation with numpy array
    input = np.array(input)
    label = np.array(label)
    
    if shuffle_color:
        channels_shuffled = np.random.permutation(3)

        input = input[:,:,channels_shuffled]
        label = label[:,:,channels_shuffled]
    
    if add_noise:
        noise = 255 * np.random.normal(mu, sigma, input.shape)

        input = input + noise
        label = label + noise

    np.clip(input, 0, 255, out=input)
    np.clip(label, 0, 255, out=label)

    input = Image.fromarray(input.astype('uint8'), 'RGB')
    label = Image.fromarray(label.astype('uint8'), 'RGB')
    
    return input, label


def get_normalize():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (1,1,1))
    ])
    return transform


class GoproDataset(data.Dataset):
    def __init__(self, root_dir, blur_type, crop_size, phase):
        self.root_dir = root_dir
        self.blur_type = blur_type
        self.crop_size = crop_size
        self.phase = phase 

        self.normalize = get_normalize()

        assert self.phase in ['train', 'test']

        if self.blur_type == 'lin':
            blur_dir = 'blur'
        elif self.blur_type == 'gamma':
            blur_dir = 'blur_gamma'
        else:
            raise ValueError('incorrect blur type given..')

        self.blur_list = glob.glob(os.path.join(root_dir, phase) + '/*/' + blur_dir + '/*.png')
        self.sharp_list = glob.glob(os.path.join(root_dir, phase) + '/*/sharp/*.png')
        assert len(self.blur_list) == len(self.sharp_list)

        print('{} dataset contains total {:d} pair of images'.format(phase, len(self.blur_list)))

    def __getitem__(self, idx):
        blur1 = Image.open(self.blur_list[idx]).convert('RGB')
        sharp1 = Image.open(self.sharp_list[idx]).convert('RGB')

        if self.phase == 'train':
            blur1, sharp1 = augment(blur1, sharp1, self.crop_size) 

        h, w = blur1.size

        blur2 = blur1.resize((h//2, w//2), Image.BICUBIC)
        sharp2 = sharp1.resize((h//2, w//2), Image.BICUBIC)

        blur3 = blur2.resize((h//4, w//4), Image.BICUBIC)
        sharp3 = sharp2.resize((h//4, w//4), Image.BICUBIC)

        blur1 = self.normalize(blur1)
        sharp1 = self.normalize(sharp1)

        blur2 = self.normalize(blur2)
        sharp2 = self.normalize(sharp2)

        blur3 = self.normalize(blur3)
        sharp3 = self.normalize(sharp3)
        
        return (blur1, blur2, blur3), (sharp1, sharp2, sharp3)

    def __len__(self):
        return len(self.blur_list)