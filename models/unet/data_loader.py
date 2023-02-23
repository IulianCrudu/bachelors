import logging
from functools import partial
from os import listdir, cpu_count
from os.path import splitext, isfile, join
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset


def load_image(filename: Path) -> Image:
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    else:
        return Image.open(filename)


def resize_image(image: Image) -> Image:
    new_size = (400, 400)
    return image.resize(new_size)


def unique_mask_values(idx: str, mask_dir: Path):
    mask_file = list(mask_dir.glob(idx + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        logging.info("Mask has 3 dimensions.")
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class Dataset(TorchDataset):
    def __init__(self, images_dir: Path, masks_dir: Path, scale: float = 1.0):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        assert 0 < scale <= 1, "Scale must be between 0 and 1"
        self.scale = scale

        self.ids = [
            splitext(file)[0] for file in listdir(images_dir)
            if isfile(join(images_dir, file))
        ]

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.masks_dir), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img: Image, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize(
            (newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC
        )
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = resize_image(load_image(mask_file[0]))
        img = resize_image(load_image(img_file[0]))

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'images': torch.as_tensor(img.copy()).float().contiguous(),
            'masks': torch.as_tensor(mask.copy()).long().contiguous()
        }


if __name__ == "__main__":
    image = Path("../../data/bdd/bdd100k/images/10/test/ac6d4f42-00000000.jpg")

    image = load_image(image)
    image = resize_image(image)
    image.show()
    # images_dir = Path("../../data/bdd/bdd100k/images/10k/train")
    # masks_dir = Path("../../data/bdd/bdd100k/labels/sem_seg/masks/train")
    #
    # train_dataset = Dataset(images_dir=images_dir, masks_dir=masks_dir)
    # train_loader = DataLoader(train_dataset, shuffle=True, **dict(batch_size=1, num_workers=cpu_count(), pin_memory=True))
    #
    # for batch in train_loader:
    #     images, masks = batch['images'], batch["masks"]
    #     print("masks", masks)
    #     break
