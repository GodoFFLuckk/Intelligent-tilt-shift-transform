import torch
from PIL import Image, ImageOps
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import ImageEnhance, Image


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


def tiltshift_color_enhance(img: Image.Image, 
                            sat: float = 1.5, 
                            contrast: float = 1.2,
                            brightness: float = 1.1,
                            sharpness: float = 1.2) -> Image.Image:
    img = ImageEnhance.Color(img).enhance(sat)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img

def infer_one_image(
    input_path: str,
    model_ckpt: str,
    save_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    input_size: int = 512
):
    net = UNet().to(device)
    checkpoint = torch.load(model_ckpt, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        net.load_state_dict(checkpoint)
    net.eval()

    img = Image.open(input_path).convert('RGB')
    orig_size = img.size

    img_tmp = img.copy()
    img_tmp.thumbnail((input_size, input_size), Image.LANCZOS)
    resized_size = img_tmp.size
    delta_w = input_size - resized_size[0]
    delta_h = input_size - resized_size[1]
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - delta_w // 2,
        delta_h - delta_h // 2
    )
    padded_img = ImageOps.expand(img_tmp, padding, fill=(0,0,0))

    to_tensor = T.ToTensor()
    tensor = to_tensor(padded_img).unsqueeze(0).to(device) 

    with torch.no_grad():
        pred = net(tensor).cpu().squeeze(0) 

    pred_cropped = TF.center_crop(pred, (resized_size[1], resized_size[0])) 
    pred_img = TF.to_pil_image(pred_cropped).resize(orig_size, Image.BICUBIC)
    pred_img = tiltshift_color_enhance(pred_img, sat=1.7, contrast=1.25, brightness=1.1, sharpness=1.1)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    pred_img.save(save_path, format='JPEG', quality=92, subsampling=0)


if __name__ == "__main__":
    infer_one_image()

