import sys
import os
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QFileDialog, QLineEdit, QFormLayout, QGroupBox, QRadioButton,
    QMessageBox, QDialog, QCheckBox
)
from dbpn import Net as DBPN

from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocusRegressorEfficientNetV2L(nn.Module):
    def __init__(self):
        super(FocusRegressorEfficientNetV2L, self).__init__()
        self.efficientnet = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)

        self.efficientnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, 2)

    def forward(self, x):
        return self.efficientnet(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX //2, diffY //2, diffY - diffY //2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class FocusRegressorUNet(nn.Module):
    def __init__(self, n_channels=1, bilinear=True):
        super(FocusRegressorUNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

focus_model_efficientnet = FocusRegressorEfficientNetV2L()
focus_model_efficientnet.load_state_dict(torch.load('foc.pth', map_location=device))
focus_model_efficientnet.eval()
focus_model_efficientnet.to(device)

focus_model_unet = FocusRegressorUNet()
focus_model_unet.load_state_dict(torch.load('focus_model_unet5.pth', map_location=device))
focus_model_unet.eval()
focus_model_unet.to(device)

upscale_model = DBPN(num_channels=3, base_filter=64, feat=256, num_stages=7, scale_factor=4)
gpus_list=range(1)
upscale_model = torch.nn.DataParallel(upscale_model, device_ids=gpus_list)
upscale_model.load_state_dict(torch.load('DBPN_x4.pth', map_location=lambda storage, loc: storage))
upscale_model.eval()
upscale_model.to(device)



class ImageLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 1px solid #aaa; background-color: #f0f0f0;")
        self._pixmap = None
        self.setMinimumSize(800, 600)

    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        self._pixmap = pixmap
        self.update_pixmap()

    def set_qimage(self, qimage):
        pixmap = QPixmap.fromImage(qimage)
        self._pixmap = pixmap
        self.update_pixmap()

    def update_pixmap(self):
        if self._pixmap:
            scaled_pixmap = self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.update_pixmap()

class ResultWindow(QDialog):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Result Image")
        self.setFixedSize(800, 600)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio))

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tilt-Shift with Depth Map")
        self.setFixedSize(1280, 720)

        self.image_path = None
        self.normalized_depth_map = None
        self.original_image = None
        self.depth_map = None
        self.transformed_image = None  

        self.image_label = ImageLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        self.transform_button = QPushButton("Transform")
        self.transform_button.clicked.connect(self.transform_image)
        self.upscale_checkbox = QCheckBox("Upscale Image")

        self.model_selection_group = QGroupBox("Select Model")
        self.efficientnet_radio = QRadioButton("EfficientNet")
        self.unet_radio = QRadioButton("UNet")
        self.efficientnet_radio.setChecked(True)
        self.efficientnet_radio.toggled.connect(self.model_changed)
        self.unet_radio.toggled.connect(self.model_changed)
        model_layout = QVBoxLayout()
        model_layout.addWidget(self.efficientnet_radio)
        model_layout.addWidget(self.unet_radio)
        self.model_selection_group.setLayout(model_layout)

        self.focus_min_edit = QLineEdit()
        self.focus_max_edit = QLineEdit()
        focus_layout = QFormLayout()
        focus_layout.addRow("Focus Min:", self.focus_min_edit)
        focus_layout.addRow("Focus Max:", self.focus_max_edit)

        right_column_layout = QVBoxLayout()
        right_column_layout.addWidget(self.load_button)
        right_column_layout.addWidget(self.model_selection_group)
        right_column_layout.addWidget(self.upscale_checkbox)
        right_column_layout.addLayout(focus_layout)
        right_column_layout.addWidget(self.transform_button)
        right_column_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label, stretch=3)
        main_layout.addLayout(right_column_layout, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def get_selected_model(self):
        if self.efficientnet_radio.isChecked():
            return "EfficientNet"
        elif self.unet_radio.isChecked():
            return "UNet"
        else:
            return None

    def load_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if file_dialog.exec():
            self.image_path = file_dialog.selectedFiles()[0]
            self.image_label.set_image(self.image_path)
            self.original_image = cv2.imread(self.image_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.depth_map = None
            self.process_depth_map(image=self.original_image)

    def process_depth_map(self, image=None):
        if image is None:
            if self.image_path:
                image = Image.open(self.image_path)
            else:
                return
        else:
            image = Image.fromarray(image.astype('uint8'), 'RGB')

        image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
        model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
        model.to(device)

        inputs = image_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        depth_map = prediction.squeeze().cpu().numpy()
        self.depth_map = depth_map
        self.normalized_depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

        self.update_focus_values()

    def model_changed(self):
        if self.image_path and self.normalized_depth_map is not None:
            self.update_focus_values()

    def update_focus_values(self):
        model_name = self.get_selected_model()
        if model_name == "EfficientNet":
            input_size = 480
            depth_map_resized = cv2.resize(self.normalized_depth_map, (input_size, input_size))
            depth_map_tensor = torch.tensor(depth_map_resized).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                focus_pred = focus_model_efficientnet(depth_map_tensor).cpu().numpy()[0]
        elif model_name == "UNet":
            input_size = 256 
            depth_map_resized = cv2.resize(self.normalized_depth_map, (input_size, input_size))
            depth_map_tensor = torch.tensor(depth_map_resized).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                focus_pred = focus_model_unet(depth_map_tensor).cpu().numpy()[0]
        else:
            focus_pred = [0.0, 1.0]

        focus_min, focus_max = focus_pred[0], focus_pred[1]

        self.focus_min_edit.setText(str(focus_min))
        self.focus_max_edit.setText(str(focus_max))

    def transform_image(self):
        if self.image_path and self.original_image is not None:
            try:
                focus_min = float(self.focus_min_edit.text())
                focus_max = float(self.focus_max_edit.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Focus Min and Focus Max must be valid numbers.")
                return

            if focus_min > focus_max:
                QMessageBox.warning(self, "Invalid Input", "Focus Min must be less than or equal to Focus Max.")
                return

            image_to_process = self.original_image.copy()

            if self.upscale_checkbox.isChecked():
                image_to_process = self.upscale_image(image_to_process)
                self.process_depth_map(image=image_to_process)

            self.apply_tilt_shift_with_depth(image_to_process, self.depth_map, focus_min, focus_max)

    def chop_forward(self,x, model, scale, shave=8, min_size=80000, nGPUs=1):
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        inputlist = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            outputlist = []
            for i in range(0, 4, nGPUs):
                with torch.no_grad():
                    input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
                with torch.no_grad():
                    output_batch = model(input_batch)
                outputlist.extend(output_batch.chunk(nGPUs, dim=0))
        else:
            outputlist = [
                self.chop_forward(patch, model, scale, shave, min_size, nGPUs) \
                for patch in inputlist]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        with torch.no_grad():
            output = Variable(x.data.new(b, c, h, w))

        output[:, :, 0:h_half, 0:w_half] \
            = outputlist[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def upscale_image(self, image):
        input_image = image.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))
        input_tensor = torch.from_numpy(input_image).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = self.chop_forward(input_tensor, upscale_model, scale=4)
            output_image = prediction.squeeze().cpu().numpy()
            output_image = np.clip(output_image, 0.0, 1.0)
            output_image = np.transpose(output_image, (1, 2, 0))
            output_image = (output_image * 255.0).astype(np.uint8)
        torch.cuda.empty_cache()
        return output_image

    def round_to_odd(self,n):
        n = int(round(n))
        return n if n % 2 == 1 else n + 1

    def apply_tilt_shift_with_depth(self, image, depth_map, focus_min, focus_max):
        depth_min = depth_map.min()
        depth_max = depth_map.max()

        mask = np.zeros_like(depth_map, dtype=np.float32)
        mask[(focus_min <= depth_map) & (depth_map <= focus_max)] = 1

        down_step = (focus_min - depth_min) / 5 if focus_min > depth_min else 0
        up_step = (depth_max - focus_max) / 5 if focus_max < depth_max else 0

        blur_masks = []
        blurred_images = []

        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * 1.3, 0, 255)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * 1.3, 0, 255)
        image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)

        for i in range(5):
            blur_mask = np.zeros_like(depth_map, dtype=np.float32)
            if down_step > 0:
                blur_mask[(focus_min - (i + 1) * down_step <= depth_map) & (depth_map < focus_min - i * down_step)] = 1
            if up_step > 0:
                blur_mask[(depth_map <= focus_max + (i + 1) * up_step) & (depth_map > focus_max + i * up_step)] = 1
            blur_mask = np.repeat(blur_mask[:, :, np.newaxis], 3, axis=2)
            blur_masks.append(blur_mask)
            kernel_size = (
                self.round_to_odd(image.shape[1] // 100 * 1.3 * (i + 1)),
                self.round_to_odd(image.shape[0] // 100 * 1.3 * (i + 1))
            )
            if kernel_size[0] < 1 or kernel_size[1] < 1:
                kernel_size = (1, 1)
            blurred_img = cv2.GaussianBlur(image, kernel_size, 0)
            blurred_images.append(blurred_img)

        mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        tilt_shift_image = image * mask_rgb
        for i in range(5):
            tilt_shift_image += blur_masks[i] * blurred_images[i]

        result_folder = 'result_images'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        base_name, ext = os.path.splitext(os.path.basename(self.image_path))
        model_name = self.get_selected_model()
        upscale_info = "_upscaled" if self.upscale_checkbox.isChecked() else ""
        output_path = os.path.join(result_folder, f"{base_name}_tilt_shift_{model_name}{upscale_info}{ext}")

        cv2.imwrite(output_path, cv2.cvtColor(tilt_shift_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
        self.show_result_image(output_path)

    def show_result_image(self, image_path):
        result_window = ResultWindow(image_path)
        result_window.exec()

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()