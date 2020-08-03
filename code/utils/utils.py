import os

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


def print_args(args):
    msg = ''
    msg += ('=' * 20 + '\n')
    msg += 'List of Arguments\n'
    msg += ('-' * 20 + '\n')
    for k, v in vars(args).items():
        msg += (str(k) + ' : ' + str(v) + '\n')
    msg += ('=' * 20)
    print(msg)


def get_summarywriter(experiment_id):
    log_path = os.path.join('./logs', experiment_id)

    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    print("saving tensorboard logs at '{}'".format(log_path))

    writer = SummaryWriter(log_path)
    assert writer is not None

    return writer    


def save_image(img, path):
    img *= 255
    np.clip(img, 0, 255, out=img)
    Image.fromarray(img.astype('uint8'), 'RGB').save(path)


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    a = get_summarywriter('temp')
    print('done')
    b = AverageMeter('ma')