# Tilt-Shift-With-Depth

## Project Overview
This application is designed to apply a Tilt-Shift effect to images, leveraging a depth map (Depth-Anything-V2-Large-hf from Hugging Face) and regression models to predict a suitable focus range. Once the focus range is identified, the system selectively blurs regions outside that range, creating the illusion of a miniature scene. Additionally, the application can upscale images before or after applying the effect, using a DBPN (Deep Back-Projection Network).

## Main Components

### 1. Depth Map Generation
Upon image loading, the application automatically generates a depth map through the `depth-anything/Depth-Anything-V2-Large-hf` model:
1. **Preprocessing**: The image is converted to RGB and packaged into a tensor.
2. **Inference**: The model produces a `predicted_depth` tensor.
3. **Postprocessing**: Depth values are interpolated to match the original resolution. A normalized depth map is stored internally for further processing.

### 2. Focus Models (Focus Range Prediction)
To determine the areas in focus, the application provides two regression models:
1. **EfficientNet-based Regressor**  
   - Modifies the first convolutional layer to handle single-channel input (the depth map).  
   - Outputs two values: `Focus Min` and `Focus Max`, indicating the depth range in focus.
2. **UNet-based Regressor**  
   - A standard UNet architecture adapted to output two regression values.  
   - Similar to the EfficientNet regressor, it predicts `Focus Min` and `Focus Max`.

When an image is loaded or a model is toggled, the depth map is passed to the selected model, and the predicted focus values are automatically shown in the GUI.  

### 3. Upscaling
For users who need higher resolution images, the DBPN (Deep Back-Projection Network) model can be optionally employed:
1. **Chop Forward**  
   - Large images are subdivided into smaller patches to avoid GPU/CPU memory overflows.  
   - Each patch is processed individually and then stitched back together.  
2. **DBPN Inference**  
   - The model upscales the patch by a factor of 4.  
   - The resulting patches are recombined to form the final upscaled image.

After upscaling, the depth map can be regenerated for the new resolution to maintain consistency in the subsequent tilt-shift step.

### 4. Tilt-Shift Transformation
With an up-to-date depth map and the predicted focus range (`focus_min`, `focus_max`), the application blurs regions beyond that focus range:
1. **Hue and Saturation Boost**  
   - The image is converted to HSV space, where saturation and value are boosted slightly for a more vibrant effect.  
2. **Blur Mask Construction**  
   - Multiple masks are generated, each corresponding to a level of blur for regions progressively farther from the focus range.  
3. **Gaussian Blur**  
   - Regions outside the focus range are blurred more heavily (using 5 progressively stronger blur levels).  
   - The final image is assembled by combining each blurred region mask with the original in-focus region.

### 5. Output
Results are saved in the `result_images` folder. The output image is displayed in a separate dialog, showing the tilt-shift effect applied with the chosen model and optional upscaling steps.

---

## How It Works

1. **Load Image**  
   - The user selects an image from the file dialog.  
   - A depth map is generated via the Hugging Face model.
2. **Select Model**  
   - Choose between **EfficientNet** or **UNet** for focus range prediction.  
   - The `focus_min` and `focus_max` values update automatically in the text fields.
3. **Configure Upscaling**  
   - Toggle the **Upscale Image** checkbox to enable DBPN-based upscaling before tilt-shift is applied.
4. **Focus Range**  
   - (Optional) Manually adjust `Focus Min` and `Focus Max`.  
   - Validation checks ensure `Focus Min <= Focus Max`.
5. **Transform**  
   - Clicking **Transform** applies the tilt-shift effect (and upscaling, if enabled).  
   - The processed image is saved to `result_images` and shown in a new window.

---

## Key Classes

1. **`FocusRegressorEfficientNetV2L`**  
   - Wraps a pretrained EfficientNet-V2-L to regress two focus values from a single-channel (depth) input.
2. **`FocusRegressorUNet`**  
   - Standard UNet, modified to output two numeric values: `focus_min` and `focus_max`.
3. **`DBPN` (located in `dbpn.py`)**  
   - Deep Back-Projection Network for upscaling images by 4×.
4. **`MainWindow`**  
   - The primary PySide6 UI class containing GUI elements for loading images, model selection, and transform execution.