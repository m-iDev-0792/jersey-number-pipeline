import cv2
import mediapipe as mp
import os
import numpy as np
import time
import json
import pywt

from tqdm import tqdm
from pathlib import Path
from datetime import timedelta
from argparse import ArgumentParser
from xtcocotools.coco import COCO
import concurrent.futures
import threading

model_complexity = 0
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=model_complexity,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

def laplacian_sr(image, scale=2):
    small = cv2.pyrDown(image)
    upsampled = cv2.pyrUp(small)
    detail = cv2.subtract(image, upsampled)
    result = cv2.add(upsampled, detail)
    return result

def wavelet_sr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coeffs2 = pywt.dwt2(gray, 'haar')
    LL, (LH, HL, HH) = coeffs2
    LH = LH * 1.5
    HL = HL * 1.5
    HH = HH * 1.5
    img_recon = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    img_recon = np.clip(img_recon, 0, 255).astype(np.uint8)
    return cv2.merge([img_recon, img_recon, img_recon])

def apply_CLAHE(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
def high_boost_filter(image, alpha=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    mask = cv2.subtract(image, blurred)
    high_boost = cv2.addWeighted(image, 1.0 + alpha, mask, -alpha, 0)
    return high_boost
def unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened
def sobel_sharpen(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.clip(sobel, 0, 255).astype(np.uint8)
    sharpened = cv2.addWeighted(image, 1.5, sobel, -0.5, 0)
    return sharpened
def resize_x4(image):
    img_size = image.shape
    result = cv2.resize(image, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_AREA)
    return result


image_enhance_algo_map = {
    'laplacian': laplacian_sr,
    'wavelet': wavelet_sr,
    'CLAHE': apply_CLAHE,
    'high_boost_filter': high_boost_filter,
    'unsharp_mask': unsharp_mask,
    'sobel_sharpen': sobel_sharpen,
    'resize_x4' : resize_x4
}


def mediapipe_to_coco(landmarks, image_width, image_height):
    mediapipe_to_coco_map = [
        0,  # Nose
        2,  # Left Eye Inner
        5,  # Right Eye Inner
        7,  # Left Ear
        8,  # Right Ear
        11,  # Left Shoulder
        12,  # Right Shoulder
        13,  # Left Elbow
        14,  # Right Elbow
        15,  # Left Wrist
        16,  # Right Wrist
        23,  # Left Hip
        24,  # Right Hip
        25,  # Left Knee
        26,  # Right Knee
        27,  # Left Ankle
        28  # Right Ankle
    ]

    coco_keypoints = []

    for idx in mediapipe_to_coco_map:
        x = landmarks[idx].x * image_width
        y = landmarks[idx].y * image_height
        visibility_flag = landmarks[idx].visibility  # 0 = not visible, 1 = visible
        # visibility_flag = 2 if visibility_flag > 0.5 else 1  # COCO uses 2 (visible), 1 (occluded), 0 (not labeled)
        coco_keypoints.append([x, y, visibility_flag])

    return coco_keypoints

def process_image(image_path, output_dir, enhanceAlgo = ''):
    start_time = time.time()
    image = cv2.imread(image_path)
    if enhanceAlgo != '' and enhanceAlgo in image_enhance_algo_map:
        image = image_enhance_algo_map[enhanceAlgo](image)
    if image is None:
        print(f"Can't read image: {image_path}")
        return None, 0

    # 转换为RGB（MediaPipe需要RGB格式）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        # print(f"[PoseNotFound] at image: {image_path}")
        # 结束计时
        process_time = time.time() - start_time
        return None, process_time

    h, w, _ = image.shape
    landmarks = results.pose_landmarks.landmark

    # 定义需要的关键点（以MediaPipe的索引为准）
    # 11: 左肩, 12: 右肩, 23: 左髋, 24: 右髋, 25: 左膝, 26: 右膝
    shoulder_l = np.array([landmarks[11].x * w, landmarks[11].y * h])
    shoulder_r = np.array([landmarks[12].x * w, landmarks[12].y * h])
    hip_l = np.array([landmarks[23].x * w, landmarks[23].y * h])
    hip_r = np.array([landmarks[24].x * w, landmarks[24].y * h])
    knee_l = np.array([landmarks[25].x * w, landmarks[25].y * h])
    knee_r = np.array([landmarks[26].x * w, landmarks[26].y * h])

    # 计算肩膀到大腿之间的区域
    top = min(shoulder_l[1], shoulder_r[1])
    bottom = max(hip_l[1], hip_r[1])
    left = min(shoulder_l[0], hip_l[0], knee_l[0])
    right = max(shoulder_r[0], hip_r[0], knee_r[0])

    # 添加一些边距
    hori_padding = 20
    vert_padding = 5
    top = max(0, top - vert_padding)
    bottom = min(h, bottom + vert_padding)
    vert = [top, bottom]
    top = max(min(vert), 0)
    bottom = max(max(vert), 0)

    left = min(max(0, left - hori_padding), w)
    right = min(w, right + hori_padding)
    hori = [left, right]
    left = int(max(min(hori), 0))
    right = int(max(hori))

    if abs(left - right) < 2 or abs(top-bottom)< 2:
        return None, 0
    if output_dir !='' and os.path.exists(output_dir):
        # 裁剪图像
        cropped_image = image[int(top):int(bottom), int(left):int(right)]

        # 保存结果
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"cropped_{base_name}")
        cv2.imwrite(output_path, cropped_image)

        # 创建骨架绘制图像
        skeleton_image = image.copy()
        skeleton_image_rgb = cv2.cvtColor(skeleton_image, cv2.COLOR_BGR2RGB)

        # 使用MediaPipe内置的绘制函数绘制骨架
        mp_drawing.draw_landmarks(
            skeleton_image_rgb,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # 转回BGR格式以便用OpenCV保存
        skeleton_image = cv2.cvtColor(skeleton_image_rgb, cv2.COLOR_RGB2BGR)

        # 在骨架图上添加裁剪区域框
        cv2.rectangle(skeleton_image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 2)

        cv2.putText(skeleton_image, "Cropping Area", (int(left), int(top) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        key_points = {
            "Left Shoulder": (int(shoulder_l[0]), int(shoulder_l[1])),
            "Right Shoulder": (int(shoulder_r[0]), int(shoulder_r[1])),
            "Left Hip": (int(hip_l[0]), int(hip_l[1])),
            "Right Hip": (int(hip_r[0]), int(hip_r[1])),
            "Left Knee": (int(knee_l[0]), int(knee_l[1])),
            "Right Knee": (int(knee_r[0]), int(knee_r[1]))
        }

        for name, (x, y) in key_points.items():
            cv2.putText(skeleton_image, name, (x + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 保存骨架图像
        skeleton_output_path = os.path.join(output_dir, f"skeleton_{base_name}")
        cv2.imwrite(skeleton_output_path, skeleton_image)

        # 创建裁剪后的骨架图像 - 仅包含裁剪区域内的骨架
        cropped_skeleton = skeleton_image[int(top):int(bottom), int(left):int(right)]
        cropped_skeleton_path = os.path.join(output_dir, f"cropped_skeleton_{base_name}")
        cv2.imwrite(cropped_skeleton_path, cropped_skeleton)
        pass # image saving ends
    process_time = time.time() - start_time
    coco_landmarks = mediapipe_to_coco(landmarks, w, h)
    return coco_landmarks, process_time


def process_directory(input_dir, output_dir):
    # 创建输出目录（如果不存在）
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

    # 遍历输入目录中的所有文件
    all_results = {}
    processing_times = {}
    total_start_time = time.time()

    # 获取符合条件的文件列表
    image_files = [file_path for file_path in Path(input_dir).glob('*')
                   if file_path.suffix.lower() in image_extensions]

    total_files = len(image_files)
    processed_files = 0

    for file_path in image_files:
        print(f"处理图像 ({processed_files + 1}/{total_files}): {file_path}")
        results, proc_time = process_image(str(file_path), output_dir)

        # 记录处理时间
        processing_times[file_path.name] = proc_time
        print(f"  - 处理时间: {proc_time:.4f} 秒")

        if results:
            all_results[file_path.name] = results
            processed_files += 1

    # 计算总处理时间和平均处理时间
    total_time = time.time() - total_start_time
    avg_time = sum(processing_times.values()) / len(processing_times) if processing_times else 0

    # 保存处理时间统计
    time_stats = {
        "total_files": total_files,
        "processed_files": processed_files,
        "total_processing_time_seconds": total_time,
        "average_processing_time_seconds": avg_time,
        "individual_times": processing_times
    }

    # 保存处理时间统计和关键点数据到文件
    import json
    with open(os.path.join(output_dir, 'landmarks.json'), 'w') as f:
        json.dump(all_results, f, indent=4)

    with open(os.path.join(output_dir, 'processing_stats.json'), 'w') as f:
        json.dump(time_stats, f, indent=4)

    # 格式化时间显示
    formatted_total = str(timedelta(seconds=round(total_time)))

    # 打印处理统计信息
    print("\nSummary:")
    print(f"Image number: {total_files}")
    print(f"Succeed image number : {processed_files}")
    print(f"Total processing time: {formatted_total} (HH:MM:SS)")
    print(f"Processing time for a single image: {avg_time:.4f} s")
    print(f"Speed: {1 / avg_time:.2f} 张/s") if avg_time > 0 else print("Speed: 0 张/s")
    print(f"\nProcessing completed。Saved to {output_dir}")
    print("\nNote:")
    print("- cropped_*.jpg: 裁剪后的图像")
    print("- skeleton_*.jpg: 包含骨架和裁剪区域的完整图像")
    print("- cropped_skeleton_*.jpg: 裁剪后的骨架图像")
    print("- landmarks.json: 所有图像的关键点数据")
    print("- processing_stats.json: 处理时间统计")

# command = f"conda run --live-stream -n {config.pose_env} python pose.py {config.pose_home}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py \
#                     {config.pose_home}/checkpoints/vitpose-h.pth --img-root / --json-file {input_json} \
#                     --out-json {output_json}"


def main():
    parser = ArgumentParser()
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--out-json',
        type=str,
        default='',
        help='Json file containing results.')
    parser.add_argument(
        '--crop-padding-horizontal',
        type=int,
        default=20,
        help='Horizontal padding for cropping.')

    parser.add_argument(
        '--thread-num',
        type=int,
        default=1,
        help='Thread pool size')

    parser.add_argument(
        '--crop-padding-vertical',
        type=int,
        default=5,
        help='Vertical padding for cropping.')

    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
             'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    args = parser.parse_args()

    coco = COCO(args.json_file)
    img_keys = list(coco.imgs.keys())

    if True:
        image = coco.loadImgs(1)[0]
        image_name = os.path.join(args.img_root, image['file_name'])
    start_time = time.time()

    def worker(image_id):
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image['file_name'])
        landmarks, _ = process_image(image_name, "")
        if not landmarks:
            landmarks = []
        result = {"img_name": image['file_name'], "id": image_id, "keypoints": landmarks}
        # print(f'predict result: {result}')
        return result

    print(f'Use thread number {args.thread_num}')
    results = []
    if args.thread_num > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.thread_num) as executor:
            results = list(executor.map(worker, img_keys))
    else:
        for image_id in tqdm(range(len(img_keys)), desc="Processing"):
            landmarks = worker(image_id)
            results.append(landmarks)
        pass

    if args.out_json != '':
        with open(args.out_json, 'w') as fp:
            print(f'media_pose.py: main() writing results to {args.out_json}...')
            json.dump({"pose_results": results}, fp)
    else:
        print(f'media_pose.py: main() there is no valid output path given!')
    end_time = time.time()
    print(f'media_pose.py: main() took {end_time - start_time:.4f} seconds')

def UnitTest():
    data_id = 1
    input_directory = f"/Users/hezhenbang/Desktop/UBCO/DL-COSC519/GroupProject/hzb-jersey-number-pipeline/data/SoccerNet/test/images/{data_id}"
    output_directory = f"/Users/hezhenbang/Desktop/UBCO/DL-COSC519/GroupProject/hzb-jersey-number-pipeline/data/SoccerNet/MediaPipe/images/{data_id}/mc_{model_complexity}"

    # input_directory = f"/Users/hezhenbang/Downloads/final_outputs/sample_frames2_sr"
    # output_directory = f"/Users/hezhenbang/Downloads/final_outputs/sample_frames2_sr_cropped"

    if os.path.exists(output_directory):
        import shutil
        shutil.rmtree(output_directory)

    process_directory(input_directory, output_directory)

if __name__ == "__main__":
    # UnitTest()
    main()