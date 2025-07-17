import os
from glob import glob
from typing import List, Tuple

from PIL import Image, ImageOps
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.optim as optim

import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure

INPUT_DIR = "./input_images"
TARGET_DIR = "./target_images"
IMG_SIZE = 512
BATCH_SIZE = 12
NUM_EPOCHS = 80
LEARNING_RATE = 3e-4
AUGMENT = True
VAL_SPLIT = 0.1
CHECKPOINT_DIR = "./checkpoints"
VISUAL_SAMPLE_DIR = "./visual_samples"
PREVIEW_DIR = "./visual_previews_all"
LOG_FILE = "train_log.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESUME_FROM = None
LOSS_L1_COEF = 0.1
LOSS_LPIPS_COEF = 1
LOSS_SSIM_COEF = 0.1

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VISUAL_SAMPLE_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

class TiltShiftDataset(Dataset):
    def __init__(self, input_dir: str, target_dir: str, img_size: int, augment: bool = True):
        self.input_paths: List[str] = sorted(glob(os.path.join(input_dir, '*')))
        self.target_paths: List[str] = sorted(glob(os.path.join(target_dir, '*')))
        assert len(self.input_paths) == len(self.target_paths), \
            'Folders must contain the same number of images'
        self.img_size = img_size
        self.augment = augment
        self.to_tensor = T.ToTensor()

    def _augment_pair(self, img1: Image.Image, img2: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if np.random.rand() < 0.5:
            img1, img2 = TF.hflip(img1), TF.hflip(img2)
        if np.random.rand() < 0.2:
            img1, img2 = TF.vflip(img1), TF.vflip(img2)
        angle = np.random.uniform(-5, 5)
        img1 = TF.rotate(img1, angle, interpolation=T.InterpolationMode.BILINEAR)
        img2 = TF.rotate(img2, angle, interpolation=T.InterpolationMode.BILINEAR)
        return img1, img2

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        inp = Image.open(self.input_paths[idx]).convert('RGB')
        tgt = Image.open(self.target_paths[idx]).convert('RGB')
        if self.augment:
            inp, tgt = self._augment_pair(inp, tgt)
        inp = self.to_tensor(inp)
        tgt = self.to_tensor(tgt)
        return inp, tgt

class VisualSampleDataset(Dataset):
    def __init__(self, folder: str, target_size: Tuple[int, int]=(512, 512)):
        IMG_EXT = {'.jpg', '.jpeg'}
        self.paths = sorted([
            p for p in glob(os.path.join(folder, '*'))
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMG_EXT
        ])
        self.target_size = target_size
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert('RGB')
        orig_size = img.size

        img_tmp = img.copy()
        img_tmp.thumbnail(self.target_size, Image.LANCZOS)
        resized_size = img_tmp.size
        delta_w = self.target_size[0] - resized_size[0]
        delta_h = self.target_size[1] - resized_size[1]
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - delta_w // 2,
            delta_h - delta_h // 2
        )
        padded_img = ImageOps.expand(img_tmp, padding, fill=(0,0,0))

        tensor = self.to_tensor(padded_img)
        filename = os.path.basename(img_path)
        return tensor, orig_size, resized_size, filename

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)

def down_block(in_ch, out_ch, dropout_p=0.0):
    return nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, dropout_p=dropout_p))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, dropout_p=0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, dropout_p=dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.inc   = DoubleConv(in_ch, 64, dropout_p=0.0)
        self.down1 = down_block(64, 128, dropout_p=0.1)
        self.down2 = down_block(128, 256, dropout_p=0.1)
        self.down3 = down_block(256, 512, dropout_p=0.2)
        self.down4 = down_block(512, 1024, dropout_p=0.3)
        self.up1 = Up(1024+512, 512, dropout_p=0.2)
        self.up2 = Up(512+256, 256, dropout_p=0.1)
        self.up3 = Up(256+128, 128, dropout_p=0.1)
        self.up4 = Up(128+64, 64, dropout_p=0.0)
        self.outc = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return torch.sigmoid(self.outc(x))

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
def ssim_loss(pred, tgt):
    return 1.0 - ssim_metric(pred, tgt)

lpips_loss_fn = lpips.LPIPS(net='vgg').to(DEVICE)

def save_visual_samples(model, visual_loader, epoch):
    model.eval()
    out_dir = os.path.join(PREVIEW_DIR, f'epoch_{epoch:03d}')
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for batch in visual_loader:
            tensors, orig_sizes, resized_sizes, filenames = batch
            tensors = torch.stack(tensors).to(DEVICE)
            preds = model(tensors).cpu()
            for pred, orig_size, resized_size, filename in zip(preds, orig_sizes, resized_sizes, filenames):
                pred_cropped = TF.center_crop(pred, (resized_size[1], resized_size[0]))
                pred_img = TF.to_pil_image(pred_cropped).resize(orig_size, Image.BICUBIC)
                out_path = os.path.join(out_dir, filename)
                pred_img.save(out_path, format='JPEG', quality=92, subsampling=0)


def train():
    full_ds = TiltShiftDataset(INPUT_DIR, TARGET_DIR, IMG_SIZE, augment=AUGMENT)
    val_len = max(1, int(len(full_ds) * VAL_SPLIT))
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    visual_ds = VisualSampleDataset(VISUAL_SAMPLE_DIR, target_size=(512, 512))
    visual_loader = DataLoader(visual_ds, batch_size=4, shuffle=False, collate_fn=lambda batch: list(zip(*batch)))

    net = UNet().to(DEVICE)
    l1_loss = nn.L1Loss()
    opt = optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    best_val_loss = float('inf')
    start_epoch = 1

    logf = open(LOG_FILE, "a")

    if RESUME_FROM is not None and os.path.isfile(RESUME_FROM):
        print(f"Loading checkpoint: {RESUME_FROM}")
        checkpoint = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            start_epoch = checkpoint.get('epoch', 1)
            print(f"Resuming from epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")
        else:
            net.load_state_dict(checkpoint)
            print("Loaded model weights only (optimizer not restored).")
    else:
        print("Starting training from scratch.")
        logf.write("epoch,train_loss,train_l1,train_ssim,train_lpips,val_loss,val_l1,val_ssim,val_lpips\n")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        net.train()
        tr_losses = []
        tr_l1 = []
        tr_ssim = []
        tr_lpips = []

        for inp, tgt in train_loader:
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            pred = net(inp)
            loss_l1 = l1_loss(pred, tgt)
            loss_ssim = ssim_loss(pred, tgt)
            loss_lpips = lpips_loss_fn(pred * 2 - 1, tgt * 2 - 1).mean()

            loss = LOSS_L1_COEF * loss_l1 + LOSS_SSIM_COEF * loss_ssim + LOSS_LPIPS_COEF * loss_lpips

            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
            tr_l1.append(loss_l1.item())
            tr_ssim.append(loss_ssim.item())
            tr_lpips.append(loss_lpips.item())

        train_loss = np.mean(tr_losses)
        train_l1 = np.mean(tr_l1)
        train_ssim = np.mean(tr_ssim)
        train_lpips = np.mean(tr_lpips)

        net.eval()
        val_losses = []
        val_l1 = []
        val_ssim = []
        val_lpips = []

        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
                pred = net(inp)
                loss_l1 = l1_loss(pred, tgt)
                loss_ssim = ssim_loss(pred, tgt)
                loss_lpips = lpips_loss_fn(pred * 2 - 1, tgt * 2 - 1).mean()
                loss = LOSS_L1_COEF * loss_l1 + LOSS_SSIM_COEF * loss_ssim + LOSS_LPIPS_COEF * loss_lpips

                val_losses.append(loss.item())
                val_l1.append(loss_l1.item())
                val_ssim.append(loss_ssim.item())
                val_lpips.append(loss_lpips.item())

        val_loss = np.mean(val_losses)
        val_l1 = np.mean(val_l1)
        val_ssim = np.mean(val_ssim)
        val_lpips = np.mean(val_lpips)


        log_line = f"{epoch},{train_loss:.4f},{train_l1:.4f},{train_ssim:.4f},{train_lpips:.4f},"\
                   f"{val_loss:.4f},{val_l1:.4f},{val_ssim:.4f},{val_lpips:.4f}\n"
        logf.write(log_line)
        logf.flush()

        print(
              f'Epoch {epoch}/{NUM_EPOCHS} | '
              f'train: {train_loss:.4f} (L1 {train_l1:.4f}, SSIM {train_ssim:.4f}, LPIPS {train_lpips:.4f}) | '
              f'val: {val_loss:.4f} (L1 {val_l1:.4f}, SSIM {val_ssim:.4f}, LPIPS {val_lpips:.4f})'
          )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(CHECKPOINT_DIR, 'unet_best.pth'))
            print(f'Best model updated at epoch {epoch} (val_loss: {val_loss:.4f})')

        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(CHECKPOINT_DIR, f'unet_epoch{epoch}.pth'))

            if len(visual_ds) > 0:
                save_visual_samples(net, visual_loader, epoch)
    logf.close()

if __name__ == "__main__":
    train()
