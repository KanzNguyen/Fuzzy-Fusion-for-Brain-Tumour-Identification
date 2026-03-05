import io
import json
import math
import joblib
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tensorflow.keras.models import load_model

BASE_DIR = {
    "Gompertz":    "D:\\iccies\\Gompertz\\",
    "Mitscherlich":"D:\\iccies\\Mitscherlich\\",
    "SVM":         "D:\\iccies\\SVM\\",
}
LABEL_MAP = {0: "No Tumor", 1: "Tumor"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_pil_image(image) -> Image.Image:
    """
    Chấp nhận:
      - str/Path  : đường dẫn file hoặc URL (http/https)
      - bytes     : raw bytes
      - file-like : object có .read()
      - PIL.Image : trả thẳng về
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str,)):
        if image.startswith("http://") or image.startswith("https://"):
            import urllib.request
            req = urllib.request.Request(
                image,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                    "Referer": image,
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        return Image.open(image).convert("RGB")

    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")

    # file-like object (e.g. Flask request.files, Django InMemoryUploadedFile…)
    if hasattr(image, "read"):
        return Image.open(image).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image)}")

def _load_mobilenet(device):
    mobilenet = models.mobilenet_v3_large(pretrained=True)
    mobilenet = nn.Sequential(*list(mobilenet.children())[:-1])
    return mobilenet.to(device).eval()


def _preprocess_tensor(pil_img: Image.Image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil_img).unsqueeze(0).to(device)


def _preprocess_pca(pil_img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(pil_img).numpy().flatten()


def _extract_mobilenet_features(pil_img, mobilenet, device):
    tensor = _preprocess_tensor(pil_img, device)
    with torch.no_grad():
        features = mobilenet(tensor).squeeze().cpu().numpy()
    return features.flatten()


def _build_feature_vector(pil_img, mobilenet, pca_model, device):
    mobilenet_features = _extract_mobilenet_features(pil_img, mobilenet, device)
    pca_raw            = _preprocess_pca(pil_img)
    pca_features       = pca_model.transform(pca_raw.reshape(1, -1)).flatten()
    return np.hstack((mobilenet_features, pca_features))


# ── Fuzzy ensemble helpers ─────────────────────────────────────────────────────
def _fuzzy_rank_gompertz(CF, top):
    R_L = 1 - np.exp(-np.exp(-2.0 * CF))
    K_L = 0.632 * np.ones(R_L.shape)
    for i in range(R_L.shape[0]):
        for sample in range(R_L.shape[1]):
            for k in range(top):
                a = R_L[i][sample]
                idx = np.where(a == np.partition(a, k)[k])
                K_L[i][sample][idx] = R_L[i][sample][idx]
    return K_L


def _fuzzy_rank_mitcherlich(CF, top):
    R_L = 2 - 1 * 2 ** CF
    K_L = 1 * np.ones(R_L.shape)
    for i in range(R_L.shape[0]):
        for sample in range(R_L.shape[1]):
            for k in range(top - 1):
                a = R_L[i][sample]
                idx = np.where(a == np.partition(a, k)[k])
                K_L[i][sample][idx] = R_L[i][sample][idx]
    return K_L


def _cfs_func(CF, K_L, neutral_val):
    H = CF.shape[0]
    for f in range(CF.shape[0]):
        for i in range(CF.shape[1]):
            idx = np.where(K_L[f][i] == neutral_val)
            CF[f][i][idx] = 0
    return 1 - np.sum(CF, axis=0) / H


def _fuzzy_ensemble(preds, method, top=2):
    """preds: list of (1, 2) arrays"""
    L  = len(preds)
    CF = np.zeros((L, preds[0].shape[0], preds[0].shape[1]))
    for i, p in enumerate(preds):
        CF[i] = p

    if method == "Gompertz":
        R_L, neutral = _fuzzy_rank_gompertz(CF, top), 0.632
    else:
        R_L, neutral = _fuzzy_rank_mitcherlich(CF, top), 1.0

    RS  = np.sum(R_L, axis=0)
    CFS = _cfs_func(CF, R_L, neutral)
    FS  = RS * CFS
    return int(np.argmin(FS, axis=1)[0])

def inference(image, method: str) -> dict:
    """
    Phân loại u não từ ảnh MRI.

    Parameters
    ----------
    image  : str | bytes | file-like | PIL.Image
             Đường dẫn file, URL (http/https), raw bytes,
             file-like object, hoặc PIL Image.
    method : str
             Một trong: "SVM" | "Gompertz" | "Mitscherlich"

    Returns
    -------
    dict với các key:
        method          (str)
        prediction      (str)   "No Tumor" hoặc "Tumor"
        predicted_class (int)   0 hoặc 1
        confidence      (float) xác suất dự đoán [0, 1]
    """
    if method not in BASE_DIR:
        raise ValueError(
            f"Unknown method: '{method}'. Choose from {list(BASE_DIR)}"
        )

    save_dir = BASE_DIR[method]
    pil_img  = _load_pil_image(image)          # chuẩn hoá đầu vào → PIL
    mobilenet = _load_mobilenet(device)

    pca_model = joblib.load(save_dir + "pca_model.joblib")
    combined  = _build_feature_vector(pil_img, mobilenet, pca_model, device)

    if method == "SVM":
        svm_model = joblib.load(save_dir + "svm_model.joblib")

        predicted_class = int(svm_model.predict(combined.reshape(1, -1))[0])
        try:
            confidence = float(
                svm_model.predict_proba(combined.reshape(1, -1))[0][predicted_class]
            )
        except AttributeError:
            score      = svm_model.decision_function(combined.reshape(1, -1))[0]
            confidence = float(1 / (1 + math.exp(-score)))

    else:  # Gompertz hoặc Mitscherlich
        model1 = load_model(save_dir + "cnn_model_32filters.keras")
        model2 = load_model(save_dir + "cnn_model_64filters.keras")
        model3 = load_model(save_dir + "cnn_model_48filters.keras")

        X_infer = combined.astype("float32").reshape(1, -1, 1)
        p1 = model1.predict(X_infer)
        p2 = model2.predict(X_infer)
        p3 = model3.predict(X_infer)

        p1_2c = np.hstack((1 - p1, p1))
        p2_2c = np.hstack((1 - p2, p2))
        p3_2c = np.hstack((1 - p3, p3))

        predicted_class = _fuzzy_ensemble([p1_2c, p2_2c, p3_2c], method, top=2)
        avg_probs       = (p1_2c + p2_2c + p3_2c) / 3
        confidence      = float(avg_probs[0][predicted_class])

    return {
        "method":          method,
        "prediction":      LABEL_MAP[predicted_class],
        "predicted_class": predicted_class,
        "confidence":      confidence,
    }
'''
if __name__ == "__main__":
    # Ví dụ 1: đường dẫn file cục bộ
    result = inference("D:\\Image1045.jpg", method="Gompertz")

    # Ví dụ 2: URL
    # result = inference("https://example.com/brain_mri.jpg", method="SVM")

    # Ví dụ 3: file-like object (Flask, Django, FastAPI…)
    # with open("D:\\Image1046.jpg", "rb") as f:
    #     result = inference(f, method="Mitscherlich")

    print(f"Method    : {result['method']}")
    print(f"Prediction: {result['prediction']} (class {result['predicted_class']})")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
'''

