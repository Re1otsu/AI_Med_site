# app/predictor.py
import os
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torchcam.methods import SmoothGradCAMpp, GradCAM

from .utils_gradcam import overlay_heatmap_on_image, cam_to_numpy

# === Пути к моделям ===
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PATH_ECG   = os.path.join(MODELS_DIR, "model_ecg.pt")
PATH_MRI   = os.path.join(MODELS_DIR, "model_mri_diagnosis.pt")
PATH_XRAY  = os.path.join(MODELS_DIR, "model_tuberculosis.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Трансформации ===
tf_ecg = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])
tf_mri = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])
tf_xray = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

# === Глобальные модели ===
_ecg_model:  Optional[nn.Module] = None
_mri_model:  Optional[nn.Module] = None
_xray_model: Optional[nn.Module] = None
_mri_classes = ["glioma", "meningioma", "pituitary", "notumor"]

# ---------- загрузчики моделей ----------
def get_ecg_model():
    global _ecg_model
    if _ecg_model is None:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 3)
        m.load_state_dict(torch.load(PATH_ECG, map_location=device))
        _ecg_model = m.to(device).eval()
    return _ecg_model

def get_mri_model():
    global _mri_model, _mri_classes
    if _mri_model is None:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 4)
        ckpt = torch.load(PATH_MRI, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            m.load_state_dict(ckpt["model_state"])
            if "classes" in ckpt:
                _mri_classes = ckpt["classes"]
        else:
            m.load_state_dict(ckpt)
        _mri_model = m.to(device).eval()
    return _mri_model

def get_xray_model():
    global _xray_model
    if _xray_model is None:
        m = models.densenet121(weights=None)  # обучалка — как у тебя
        num_ftrs = m.classifier.in_features
        m.classifier = nn.Linear(num_ftrs, 1)
        m.load_state_dict(torch.load(PATH_XRAY, map_location=device))
        _xray_model = m.to(device).eval()
    return _xray_model

# ---------- автоопределение модальности ----------
def detect_type(pil_img: Image.Image) -> str:
    conf = {}
    try:
        ecg = get_ecg_model()
        x = tf_ecg(pil_img).unsqueeze(0).to(device)
        with torch.no_grad(): conf["ecg"] = float(torch.softmax(ecg(x), dim=1)[0].max().item())
    except: conf["ecg"] = 0.0
    try:
        mri = get_mri_model()
        x = tf_mri(pil_img).unsqueeze(0).to(device)
        with torch.no_grad(): conf["mri"] = float(torch.softmax(mri(x), dim=1)[0].max().item())
    except: conf["mri"] = 0.0
    try:
        xray = get_xray_model()
        x = tf_xray(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.sigmoid(xray(x)).item()
            conf["xray"] = float(max(p, 1 - p))
    except: conf["xray"] = 0.0
    return max(conf, key=conf.get)

# ---------- ЭКГ ----------
def predict_ecg(pil_img: Image.Image, save_heatmap_path: Optional[str]) -> Dict[str, Any]:
    classes = ["Arrhythmia", "Critical", "Normal"]
    model = get_ecg_model()
    x = tf_ecg(pil_img).unsqueeze(0).to(device)

    # 1) прямой проход ДЛЯ CAM (без no_grad!)
    cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")
    out = model(x)
    probs = torch.softmax(out, dim=1)[0]
    cls_idx = int(torch.argmax(probs).item())
    prob = float(probs[cls_idx].detach().cpu().item() * 100)

    heatmap_path = None
    if save_heatmap_path:
        cams = cam_extractor(cls_idx, out)
        hm = cam_to_numpy(cams)
        overlay = overlay_heatmap_on_image(pil_img, hm, (256, 256), alpha=0.5)
        import cv2; cv2.imwrite(save_heatmap_path, overlay)
        heatmap_path = save_heatmap_path

    label = classes[cls_idx]
    diagnosis = {
        "Normal":     "✅ Ритм сердца в пределах нормы.",
        "Arrhythmia": "⚠️ Признаки аритмии. Рекомендуется консультация кардиолога.",
        "Critical":   "🚨 Критические изменения миокарда. Требуется срочная помощь."
    }[label]

    return {"modality":"ECG","label":label,"probability":round(prob,2),"diagnosis":diagnosis,"heatmap_path":heatmap_path}

# ---------- МРТ ----------
def predict_mri(pil_img: Image.Image, save_heatmap_path: Optional[str]) -> Dict[str, Any]:
    model = get_mri_model()
    x = tf_mri(pil_img).unsqueeze(0).to(device)

    cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")
    out = model(x)
    probs = torch.softmax(out, dim=1)[0]
    cls_idx = int(torch.argmax(probs).item())
    prob = float(probs[cls_idx].detach().cpu().item() * 100)

    classes = list(_mri_classes)
    label = classes[cls_idx] if 0 <= cls_idx < len(classes) else "unknown"

    heatmap_path = None
    if save_heatmap_path:
        cams = cam_extractor(cls_idx, out)
        hm = cam_to_numpy(cams)
        overlay = overlay_heatmap_on_image(pil_img, hm, (224, 224), alpha=0.5)
        import cv2; cv2.imwrite(save_heatmap_path, overlay)
        heatmap_path = save_heatmap_path

    diagnosis_map = {
        "glioma":     "🧬 Глиома — вероятно злокачественное образование.",
        "meningioma": "🧫 Менингиома — чаще доброкачественная, требуется наблюдение.",
        "pituitary":  "🧠 Опухоль гипофиза — возможны эндокринные нарушения.",
        "notumor":    "✅ Признаков опухоли не выявлено."
    }
    diagnosis = diagnosis_map.get(label, "Описание недоступно.")

    return {"modality":"MRI","label":label,"probability":round(prob,2),"diagnosis":diagnosis,"heatmap_path":heatmap_path}

# ---------- ФЛГ (X-ray) с Grad-CAM ----------
def predict_xray(pil_img: Image.Image, save_heatmap_path: Optional[str]) -> Dict[str, Any]:
    model = get_xray_model()
    x = tf_xray(pil_img).unsqueeze(0).to(device)

    # ВАЖНО: для CAM — делаем forward без no_grad
    # Таргет-слой для DenseNet121 — берём поздний conv:
    # подойдут: "features.denseblock4.denselayer16.conv2" или "features.norm5" (но conv даёт картинку детальнее)
    target_layer = "features.denseblock4.denselayer16.conv2"
    cam_extractor = GradCAM(model, target_layer=target_layer)

    out = model(x)                                # forward для CAM
    p = torch.sigmoid(out).detach().cpu().item()  # вероятность патологии

    # Уровень риска
    if p >= 0.85:
        label = "🚨 Критично"
        diagnosis = "Высокая вероятность патологии. Требуется немедленная консультация."
        risk_level = "high"
    elif p >= 0.5:
        label = "⚠️ Подозрительно"
        diagnosis = "Обнаружены изменения. Рекомендуется дополнительная проверка."
        risk_level = "medium"
    else:
        label = "✅ Вероятно норма"
        diagnosis = "Признаков патологии не выявлено."
        risk_level = "low"

    heatmap_path = None
    if save_heatmap_path:
        cams = cam_extractor(class_idx=0, scores=out)  # бинарная задача — class_idx=0
        hm = cam_to_numpy(cams)
        overlay = overlay_heatmap_on_image(pil_img, hm, (320, 320), alpha=0.5)
        import cv2; cv2.imwrite(save_heatmap_path, overlay)
        heatmap_path = save_heatmap_path

    return {
        "modality":"X-ray",
        "label":label,
        "probability":round(p*100, 2),
        "diagnosis":diagnosis,
        "risk_level":risk_level,
        "heatmap_path":heatmap_path
    }

# ---------- универсальный маршрутизатор ----------
def predict_image(pil_img: Image.Image, workdir: str = "."):
    modality = detect_type(pil_img)

    if modality == "ecg":
        result = predict_ecg(pil_img, os.path.join(workdir, "ecg_gradcam.png"))
        summary = f"ЭКГ → {result['label']} ({result['probability']}%) — {result['diagnosis']}"
    elif modality == "mri":
        result = predict_mri(pil_img, os.path.join(workdir, "mri_gradcam.png"))
        pretty = {"glioma":"Глиома","meningioma":"Менингиома","pituitary":"Опухоль гипофиза","notumor":"Без признаков опухоли"}.get(result["label"], result["label"])
        summary = f"МРТ → {pretty} ({result['probability']}%) — {result['diagnosis']}"
    else:
        result = predict_xray(pil_img, os.path.join(workdir, "xray_gradcam.png"))
        summary = f"Флюорография → {result['label']} ({result['probability']}%) — {result['diagnosis']}"

    return summary, result.get("heatmap_path"), result
