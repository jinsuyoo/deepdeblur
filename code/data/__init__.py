import torch.utils.data as data

from data import gopro_dataset


def create_dataloader(args, phase, inf=False):
    crop_size = args.crop_size if phase == 'train' else -1
    batch_size = args.batch_size if phase == 'train' else 1

    dataset = gopro_dataset.GoproDataset(
        root_dir=args.dataset, 
        blur_type=args.blur_type,
        crop_size=crop_size,
        phase=phase)
    
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=args.n_cpu,
        drop_last=True if phase == 'train' else False)

    if inf:
        return inf_data_gen(dataloader)
    else:
        return dataloader


def inf_data_gen(loader):
    while True:
        for input, label in loader:
            yield (input, label) 