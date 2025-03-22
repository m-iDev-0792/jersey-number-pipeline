import os
import io
import json
import random
import sys

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
import torch.nn as nn
from argparse import ArgumentParser

from tqdm import tqdm


class JerseyNumberClassifier(nn.Module):
    def __init__(self):
        super(JerseyNumberClassifier, self).__init__()

        # 64x64 input
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 32x32 inputs
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 16x16 inputs
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 8x8 inputs
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 4x4 inputs
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 2x2 inputs
        self.block6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 1x1 inputs
        self.fc = nn.Sequential(
            nn.Linear(1024 * 1 * 1, 8192),  # output of last conv block is 1024 * 1 * 1
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 100)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

#

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(64),      # Resize the smaller edge to 64 while keeping aspect ratio
    transforms.CenterCrop(64),  # Ensures final size is 64x64
    transforms.ToTensor()       # Convert to PyTorch tensor (C, H, W)
])

def predict_single_image(img_path, model):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cpu()
    # forward pass
    outputs = model(img_tensor)
    # get prediction
    softmax_output = torch.nn.functional.softmax(outputs, dim=1)
    max_prob, max_index = torch.max(softmax_output, dim=1)
    final_class = max_index.item()
    probability = max_prob.item()
    # print(f"image: {img_path}, final_class: {final_class}, probability: {probability}")
    return final_class, probability

def UnitTest():
    # model = torch.load('/Users/hezhenbang/Downloads/cnn_numrecognition2.pth')
    model = torch.load('/Users/hezhenbang/Downloads/cnn_numrecognition_colab.pth', map_location=torch.device('cpu'))
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    model.eval()
    image_dir = '/Users/hezhenbang/Desktop/UBCO/DL-COSC519/GroupProject/Code/jersey-2023/MediaPipe/images/0/cropped2'
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        if image_name.endswith(".jpg") or image_name.endswith(".png"):
            predict_single_image(image_path, model)

def run_inference(model, data_root, result_file):
    # load images one by one, save paths and result
    file_dir = os.path.join(data_root, 'imgs')
    filenames = os.listdir(file_dir)
    filenames.sort()
    results = {}
    for filename in tqdm(filenames):
        if not filename.endswith(".jpg") and not filename.endswith(".png"):
            continue
        image_path = os.path.join(file_dir, filename)
        confidence, prediction = predict_single_image(image_path, model)
        results[filename] = {'label':f'{prediction}', 'confidence':confidence, 'raw': [], 'logits':[]}
    with open(result_file, 'w') as f:
        json.dump(results, f)

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data', help='Dataset root directory')
    parser.add_argument('--result_file', type=str, default='outputs/preds.json' , help='Path of output json file')
    parser.add_argument('--model_path', type=str, default='data', help='Dataset root directory')

    args = parser.parse_args()
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f'model path {model_path} does not exist')
        sys.exit(1)
    print(f'loading model from path: {model_path}')
    model = None
    if torch.cuda.is_available():
        print(f'load model in gpu mode')
        model = torch.load(model_path)
        model.cuda()
    else:
        print(f'load model in cpu mode')
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.cpu()
    model.eval()
    run_inference(model, args.data_root, args.result_file)



if __name__ == '__main__':
    main()
