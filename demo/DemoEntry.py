import shutil

import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time
import os
import threading
import json
import platform
import cv2
import mediapipe as mp
import numpy as np
import configuration

segmentation = None
pose = None

G_OUTPUT_DIR="../data/SoccerNet/demo/images/0"

def crop_human(image_path, output_path):
    if 'DS_Store' in image_path:
        return None
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Initialize MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = segmenter.process(image_rgb)

        # Get segmentation mask
        mask = results.segmentation_mask
        binary_mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary (0 or 255)

        # Find contours and get bounding box
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No human detected.")
            return None

        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))  # Get largest contour (person)

        # Crop the image
        cropped_image = image[y:y + h, x:x + w]

        # Save and return the cropped image
        cv2.imwrite(output_path, cropped_image)
        print(f"Cropped image saved as {output_path}")
        return cropped_image

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

def ApplyESRGan():
    if configuration.pose_detection_pipeline!='openpose' and configuration.pose_detection_pipeline!='OpenPose':
        return
    if not os.path.exists('../openpose'):
        print(f'ApplyESRGan(): OpenPose not found, skipping...')
        return
    output_dir = "../data/SoccerNet/demo/images"
    input_dir = "../data/SoccerNet/demo/temp"
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
        cmd = f'..\\realesrgan\\realesrgan-ncnn-vulkan.exe -i {subdir_path} -o {outsub_dir} -s 2 -f jpg'
        print(f'Running command: {cmd}')
        os.system(cmd)


def CheckRunResult(final_result_path = 'out/SoccerNetResult/demo_final_results.json'):
    result = {'0':'-1'}
    result_txt = ''
    if os.path.exists(final_result_path):
        with open(final_result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
            result_txt = ''
    else:
        print(f'CheckRunResult(): {final_result_path} does not exist!')
    for item in result:
        result_txt += f'Prediction is {result[item]}\n'
        break
    root = tk.Tk()
    root.title("Prediction")
    root.geometry("800x400")  # 设置窗口大小

    # 创建标签并设置大号字体和红色
    label = tk.Label(root, text=result_txt, font=("Arial", 80, "bold"), fg="red")
    label.pack(expand=True)
    root.mainloop()


def CallPipeline():
    os.chdir('../')
    if os.path.exists(final_result_path):
        print(f'CallPipeline(): {final_result_path} exists, remove it now')
        os.remove(final_result_path)
    print(f'CallPipeline(): current working directory = {os.getcwd()}')
    os.system('echo current working directory')
    if platform.system() == 'Windows':
        os.system('dir')
    else:
        os.system('pwd')
    command = f'conda run --live-stream -n SoccerNet python main.py SoccerNet demo'
    # os.system(command)


    pass

class VideoRecorderApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Video Recorder")
        self.sample_interval = 16 #ms

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        self.record_btn = tk.Button(window, text="Record", command=self.toggle_recording, font=("Arial", 30, "bold"))
        self.record_btn.pack(pady=10)

        self.recording = False
        self.out = None
        self.countdown = 30
        self.countdown_label = tk.Label(window, text="")
        self.countdown_label.pack()

        # 创建输出文件夹
        self.output_folder = G_OUTPUT_DIR
        if os.path.exists(self.output_folder):
            print(f'Output folder {self.output_folder} already exists! Now clean it')
            shutil.rmtree(self.output_folder)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.update_preview()

    def update_preview(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        # update every self.sample_interval
        self.window.after(self.sample_interval, self.update_preview)

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recording = True
        self.record_btn.config(text="Stop", font=("Arial", 30, "bold"))

        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.video_file = f"DemoRecord_{timestamp}.avi"
        self.out = cv2.VideoWriter(self.video_file, fourcc, 20.0, (self.width, self.height))

        # 启动录制线程
        self.countdown = 30
        self.thread = threading.Thread(target=self.record_video)
        self.thread.start()
        self.update_countdown()

    def record_video(self):
        start_time = time.time()
        while self.recording and (time.time() - start_time) < 30:
            ret, frame = self.cap.read()
            if ret and self.out:
                self.out.write(frame)
            time.sleep(0.01)

        if self.recording:  # 自动结束
            self.stop_recording()

    def update_countdown(self):
        if self.recording:
            self.countdown_label.config(text=f"Remaining time: {self.countdown}s", font=("Arial", 30, "bold"))
            self.countdown -= 1
            if self.countdown >= 0:
                self.window.after(1000, self.update_countdown)
            else:
                self.stop_recording()

    def stop_recording(self):
        self.recording = False
        self.record_btn.config(text="Record", font=("Arial", 30, "bold"))
        self.countdown_label.config(text="Record finished!", font=("Arial", 30, "bold"))

        if self.out:
            self.out.release()
            self.out = None

        self.extract_frames()
        #todo. call pipeline here
        ApplyESRGan()
        CallPipeline()
        CheckRunResult()

    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_file)
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / 5)  # 每秒5帧

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(self.output_folder, f"0_{frame_count:06d}.jpg")
                frame = crop_human_from_loaded_img(frame)
                if frame is not None:
                    cv2.imwrite(frame_filename, frame)

            frame_count += 1

        cap.release()
        print(f"Sampled video into images, exported to {self.output_folder}")

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        if self.out:
            self.out.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoRecorderApp(root)
    root.mainloop()
    # gap = 50.8 / 31
    # for i in range(0, 31):
    #     v = gap * (i+1)
    #     print(f'value for {i+1} = {v}')