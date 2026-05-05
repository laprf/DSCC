import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from osgeo import gdal
import torch.nn.functional as F
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
    '.ppm', '.PPM', '.bmp', '.BMP', '.tif'
]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt):
        for t in self.transforms:
            img, gt = t(img, gt)
        return img, gt


class RandomHorizontallyFlip(object):
    def __call__(self, img, gt):
        if random.random() < 0.5:
            return (
                img[:, :, torch.arange(img.shape[2] - 1, -1, -1)],
                gt[:, torch.arange(gt.shape[1] - 1, -1, -1)]
            )
        else:
            return img, gt


class RandomCrop(object):
    def __call__(self, img, gt):
        gt = torch.tensor(gt) if not isinstance(gt, torch.Tensor) else gt
        H, W = gt.shape
        randw = torch.randint(W // 8 + 1, (1,)).item()
        randh = torch.randint(H // 8 + 1, (1,)).item()
        offseth = 0 if randh == 0 else torch.randint(randh, (1,)).item()
        offsetw = 0 if randw == 0 else torch.randint(randw, (1,)).item()
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        img = img[:, p0:p1, p2:p3]
        gt = gt[p0:p1, p2:p3]
        return img, gt

class WHU_OHS_Dataset(Dataset):
    def __init__(self, image_file_list, label_file_list, img_size, transform=None):
        self.image_file_list = image_file_list
        self.label_file_list = label_file_list
        self.img_size = img_size
        self.transform = transform

    def sample_stat(self):
        """Statistics of samples of each class in the dataset."""
        sample_per_class = torch.zeros([24])
        for label_file in self.label_file_list:
            label = gdal.Open(label_file, gdal.GA_ReadOnly)
            label = label.ReadAsArray()
            count = np.bincount(label.ravel(), minlength=25)
            count = count[1:25]
            count = torch.tensor(count)
            sample_per_class += count

        return sample_per_class

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):
        image_file = self.image_file_list[index]
        label_file = self.label_file_list[index]
        name = os.path.basename(image_file)
        image_dataset = gdal.Open(image_file, gdal.GA_ReadOnly)
        label_dataset = gdal.Open(label_file, gdal.GA_ReadOnly)

        image = image_dataset.ReadAsArray()
        label = label_dataset.ReadAsArray()
        image = torch.tensor(image, dtype=torch.float) / 10000.0
        label = torch.tensor(label, dtype=torch.float)

        if self.transform is not None:
            image, label = self.transform(image, label)

        if self.img_size != 512:
            img_size = self.img_size
            image = F.interpolate(image.unsqueeze(0), size=(img_size, img_size), mode='nearest').squeeze(0)
            label = F.interpolate(label.unsqueeze(0).unsqueeze(0), size=(img_size, img_size), mode='nearest').squeeze(0).squeeze(0)
        
        return image, label - 1.0, name.split(".")[0]

def get_dataset_loader(mode, split_dir, data_path, batch_size, img_size, shuffle,
                       transform=None, num_workers=8, prefetch_factor=4,
                       persistent_workers=None):
    image_list = []
    label_list = []
    txt_file_path = os.path.join(split_dir, mode + '.txt')

    if not os.path.exists(txt_file_path):
        raise FileNotFoundError(
            f"Split file not found for mode '{mode}': {txt_file_path}. "
            f"Set --split_dir to the directory containing {mode}.txt."
        )

    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(',')
            image_path = os.path.join(data_path, line[0] + '.tif')
            label_path = os.path.join(data_path.replace('image', 'label'), line[0] + '.tif')

            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found: {label_path}")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image_list.append(image_path)
            label_list.append(label_path)

    assert len(image_list) == len(label_list), "The number of images and labels must be equal!"

    dataset = WHU_OHS_Dataset(
        image_file_list=image_list,
        label_file_list=label_list,
        img_size=img_size,
        transform=transform
    )
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': True
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
        if persistent_workers is not None:
            loader_kwargs['persistent_workers'] = persistent_workers

    loader = DataLoader(dataset, **loader_kwargs)
    return loader


def load_data(args, img_size, mode='tr'):
    assert mode in ['tr', 'val', 'ts'], "Invalid mode. Mode should be either 'tr', 'val' or 'ts'."
    data_path = os.path.join(args.data_root, mode, 'image')
    is_shuffle = True if mode == 'tr' else False
    default_split_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'txts')
    split_dir = getattr(args, 'split_dir', default_split_dir)
    num_workers = getattr(args, 'num_workers', 8)
    prefetch_factor = getattr(args, 'prefetch_factor', 4)
    persistent_workers = getattr(args, 'persistent_workers', True)
    transform = Compose([
        RandomHorizontallyFlip(),
        RandomCrop()
    ]) if mode == 'tr' else None
    loader = get_dataset_loader(
        mode,
        split_dir,
        data_path,
        args.batch_size,
        img_size,
        is_shuffle,
        transform=transform,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
    return loader
