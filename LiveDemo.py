import os,sys,json
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time
import os
import shutil
import threading
import json
import platform
import cv2
import mediapipe as mp
import numpy as np
import configuration
import helpers

def CreateLegibleJson():
    output_json = r'out\SoccerNetResults\demo_legible.json'

    image_folder = './data/SoccerNet/demo/images/0'

    obj = {
        "0":[]
    }
    for filename in os.listdir(image_folder):
        filename = os.path.join(image_folder, filename)
        if filename.endswith('.jpg') or filename.endswith('.png'):
            obj["0"].append(filename)

    with open(output_json, 'w') as f:
        json.dump(obj, f, indent=4)



def crop_human_from_loaded_img(image):
    h, w, _ = image.shape
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = segmenter.process(image_rgb)
        mask = results.segmentation_mask
        binary_mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary (0 or 255)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No human detected.")
            return None
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))  # Get largest contour (person)
        cropped_image = image[y:y + h, x:x + w]
        return cropped_image

def resize_to_width(image, target_width=64):
    # Get original dimensions
    original_height, original_width = image.shape[:2]

    # Compute new height to maintain aspect ratio
    aspect_ratio = original_height / original_width
    new_height = int(target_width * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def extract_frames(video_path, output_folder, sample_rate=10):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            frame_filename = os.path.join(output_folder, f"0_{saved_count:05d}.jpg")
            frame = crop_human_from_loaded_img(frame)
            frame = resize_to_width(frame)
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done: {saved_count} frames saved to '{output_folder}'.")

def delete_files_with_prefix(folder_path, prefix):
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")



def ApplyESRGan():
    if configuration.pose_detection_pipeline!='openpose' and configuration.pose_detection_pipeline!='OpenPose':
        return
    if not os.path.exists('openpose'):
        print(f'ApplyESRGan(): OpenPose not found, skipping...')
        return
    output_dir = "data/SoccerNet/demo/images"
    input_dir = "data/SoccerNet/demo/temp"
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir)

    for subdir in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir)
        dst_path = os.path.join(input_dir, subdir)
        print(f'ApplyESRGan(): move {subdir_path} to {dst_path}')
        shutil.move(subdir_path, dst_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        outsub_dir = os.path.join(output_dir, subdir)
        if not os.path.exists(outsub_dir):
            os.makedirs(outsub_dir)
        cmd = f'realesrgan\\realesrgan-ncnn-vulkan.exe -i {subdir_path} -o {outsub_dir} -s 2 -f jpg'
        print(f'Running command: {cmd}')
        os.system(cmd)

demo_img_path = r".\data\SoccerNet\demo\images\0"
extract_frames(r"C:\Users\user\Downloads\send_to_Zhenbang\demo\DemoRecord_20250407_183851.avi", demo_img_path, 3)
helpers.show_images_from_folder(demo_img_path, 5, "extracted")

# ApplyESRGan()
# helpers.show_images_from_folder(demo_img_path, 5, "SR")

print(f'delete out/SoccerNetResults/demo')
shutil.rmtree('out/SoccerNetResults/demo', ignore_errors=True)
print(f'delete out/SoccerNetResults/demo_crops')
shutil.rmtree('out/SoccerNetResults/demo_crops', ignore_errors=True)
print(f'delete openpose/openpose-cache')
shutil.rmtree('openpose/openpose-cache', ignore_errors=True)
delete_files_with_prefix('out/SoccerNetResults', "demo")
os.system('conda run -n SoccerNet --live-stream python main.py SoccerNet demo')
# helpers.check_run_result()