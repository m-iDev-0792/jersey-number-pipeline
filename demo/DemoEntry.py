import shutil

import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time
import os
import threading


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
        self.output_folder = "../data/SoccerNet/demo/images/0"
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
                frame_filename = os.path.join(self.output_folder, f"frame_{frame_count:06d}.jpg")
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