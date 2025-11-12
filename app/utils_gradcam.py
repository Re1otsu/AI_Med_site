# app/utils_gradcam.py
import numpy as np
import cv2
from PIL import Image

def cam_to_numpy(cams):
    """
    Универсально приводит вывод torchcam к np.ndarray (H, W),
    работает и для torchcam<=0.3.x, и для >=0.4.0.
    """
    import torch

    # torchcam иногда возвращает: Tensor | [Tensor] | [[Tensor]]
    x = cams
    if isinstance(x, (list, tuple)):
        x = x[0]
        if isinstance(x, (list, tuple)):
            x = x[0]

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.nan_to_num(x)

    # Сжать до 2D
    if x.ndim == 3:
        x = x.mean(axis=0)
    elif x.ndim == 4:
        x = x.mean(axis=(0, 1))

    # Нормировка 0..1
    x = x - x.min()
    den = (x.max() - x.min()) + 1e-8
    x = x / den
    return x

def overlay_heatmap_on_image(pil_img: Image.Image, heatmap_01: np.ndarray, target_size=None, alpha=0.5):
    """
    Наложение тепловой карты на изображение (OpenCV colormap).
    heatmap_01 — массив 0..1 (H,W).
    """
    img = pil_img.convert("RGB")
    if target_size is not None:
        img = img.resize(target_size)

    img_cv = np.array(img)[:, :, ::-1]  # RGB->BGR
    h, w = img_cv.shape[:2]
    hm = cv2.resize(heatmap_01, (w, h))
    hm_u8 = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_cv, 1 - alpha, hm_color, alpha, 0)
    return overlay  # BGR
