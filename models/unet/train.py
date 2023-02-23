import os
import logging
from pathlib import Path

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm

from .evaluate import evaluate
from .model import UNet
from .data_loader import Dataset
from .dice_score import dice_loss

images_dir = Path("data/bdd/bdd100k/images/10/test")
masks_dir = Path("data/bdd/bdd100k/images/10/masks")
# masks_dir = Path("data/bdd/bdd100k/labels/sem_seg/masks/train")
checkpoints_dir = Path('./checkpoints/')

val_images_dir = Path("data/bdd/bdd100k/images/10k/val")
val_masks_dir = Path("data/bdd/bdd100k/labels/sem_seg/masks/val")

writer = SummaryWriter()


def train_model(
    model: UNet,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
):
    # 1. Create datasets
    train_dataset = Dataset(images_dir=images_dir, masks_dir=masks_dir)
    val_dataset = Dataset(images_dir=val_images_dir, masks_dir=val_masks_dir)

    # 2. Create data loader
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
    ''')

    # 3. Set up optimizer, loss function, and the learning rate scheduler
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=5
    )  # goal: maximize Dice score
    criterion = nn.CrossEntropyLoss() if model.out_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 4. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        with tqdm(
            total=len(train_dataset),
            desc=f"Epoch {epoch}/{epochs}",
            unit='img'
        ) as pbar:
            for index, batch in enumerate(train_loader):
                images, true_masks = batch['images'], batch['masks']

                assert images.shape[1] == model.in_channels

                images = images.to(dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(dtype=torch.long)

                masks_pred = model(images)
                if model.out_classes == 1:
                    loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                else:
                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, model.out_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                logging.info({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation
                division_step = (len(train_dataset) // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_score = evaluate(model, val_loader, device, False)
                        scheduler.step(val_score)

        Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        state_dict["mask_values"] = train_dataset.mask_values
        torch.save(state_dict, str(checkpoints_dir / 'checkpoint_epoch{}.pth'.format(epoch)))
        logging.info(f"Checkpoint {epoch} saved!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cpu')
    model = UNet(in_channels=3, out_classes=20)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.in_channels} input channels\n'
                 f'\t{model.out_classes} output channels (classes)\n'
                 f'\t"Transposed conv upscalingl')

    model.to(device=device)
    train_model(
        model=model,
        device=device,
    )
