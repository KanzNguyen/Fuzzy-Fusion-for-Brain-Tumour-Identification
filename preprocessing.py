import cv2
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score
import joblib
import torch
import torch.nn as nn
from torchvision.transforms import Resize, ToTensor, Compose, Normalize, transforms
from torchvision.models import resnet34, resnet18
from tqdm import tqdm
import imutils
import albumentations as A

def preprocess_image(img_path, output_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[SKIP] Cannot read: {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 23, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        print(f"[SKIP] No contours found: {img_path}")
        return

    l, r, t, b = [], [], [], []
    for contour in cnts:
        l.append(tuple(contour[contour[:, :, 0].argmin()][0]))
        r.append(tuple(contour[contour[:, :, 0].argmax()][0]))
        t.append(tuple(contour[contour[:, :, 1].argmin()][0]))
        b.append(tuple(contour[contour[:, :, 1].argmax()][0]))

    leftmost  = min(l, key=lambda p: p[0])
    rightmost = max(r, key=lambda p: p[0])
    topmost   = min(t, key=lambda p: p[1])
    bottommost= max(b, key=lambda p: p[1])

    ADD_PIXELS = 0
    new_img = img[
        topmost[1]   - ADD_PIXELS : bottommost[1] + ADD_PIXELS,
        leftmost[0]  - ADD_PIXELS : rightmost[0]  + ADD_PIXELS
    ].copy()

    if new_img.size == 0:
        print(f"[SKIP] Crop is empty: {img_path}")
        return

    cv2.imwrite(output_path, new_img)


def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]

    if len(image_files) == 0:
        print("No image files found in input folder.")
        return

    print(f"Found {len(image_files)} image(s). Processing...")

    for filename in tqdm(image_files):
        input_path  = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)  # giữ nguyên tên
        preprocess_image(input_path, output_path)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch preprocess brain tumor images.")
    parser.add_argument("input_folder",  type=str, help="Path to folder containing raw images")
    parser.add_argument("output_folder", type=str, help="Path to folder for preprocessed images")
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder)
