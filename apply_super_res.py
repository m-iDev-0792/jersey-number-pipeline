import os,sys,time
from argparse import ArgumentParser

def apply_super_res_per_folder(directory, out_directory):
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            out_path = os.path.join(out_directory, item)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            if os.path.isdir(item_path):
                start_time = time.time()
                print(f"Applying super resolution on folder: {item_path}")
                cmd = f"conda run --live-stream -n mediapose python inference_realesrgan.py -n RealESRGAN_x4plus -i {item_path} -o {out_path}"
                print(f'Running command: {cmd}')
                os.system(cmd)
                end_time = time.time()
                print(f"Finished in {end_time - start_time} seconds.")
                #sys.exit(0)
    except FileNotFoundError:
        print(f"Directory not found: {directory}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='./data/SoccerNet/', help='Dataset root directory')
    parser.add_argument('--dataset-category', type=str, default='test', help='The category of the dataset')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    category = args.dataset_category

    image_path = os.path.join(dataset_path, f'{category}/images/')
    out_path = os.path.join(dataset_path, f'{category}_sr/images/')

    print(f'Applying Super Resolution using ESRGAN')
    print(f'dataset category: {category}')
    print(f'image path: {image_path}')
    print(f'output path: {out_path}')

    os.chdir('./SuperRes/Real-ESRGAN');

    start_time = time.time()
    apply_super_res_per_folder(image_path, out_path)
    end_time = time.time()
    print(f"Super Resolution finished in {end_time - start_time} seconds.")
