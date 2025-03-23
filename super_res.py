import onnxruntime as ort
import numpy as np
import cv2
import pywt

def Test():
    session = ort.InferenceSession("super_resolution.onnx")
    img = cv2.imread("/Users/hezhenbang/Desktop/UBCO/DL-COSC519/GroupProject/Code/jersey-2023/test/images/0/0_2.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    output = session.run(None, {"input": img})[0]
    output = np.squeeze(output).transpose(1, 2, 0)
    output = (output * 255).astype(np.uint8)
    cv2.imwrite(
        "/Users/hezhenbang/Desktop/UBCO/DL-COSC519/GroupProject/Code/jersey-2023/test/images/0/0_2_super_resed.jpg",
        cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

def laplacian_sr(image, scale=2):
    """Laplacian 金字塔超分辨率"""
    # 降采样
    small = cv2.pyrDown(image)

    # 上采样
    upsampled = cv2.pyrUp(small)

    # 计算细节残差
    detail = cv2.subtract(image, upsampled)

    # 将细节加回到上采样图像
    result = cv2.add(upsampled, detail)

    return result

def wavelet_sr(image):
    """使用小波变换进行超分辨率"""
    # 将图像转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 进行小波变换
    coeffs2 = pywt.dwt2(gray, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # 增强高频分量
    LH = LH * 1.5
    HL = HL * 1.5
    HH = HH * 1.5

    # 逆变换重构
    img_recon = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    img_recon = np.clip(img_recon, 0, 255).astype(np.uint8)

    return cv2.merge([img_recon, img_recon, img_recon])

def apply_CLAHE(image):
    """对 YUV 颜色空间中的亮度通道应用 CLAHE 增强"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def high_boost_filter(image, alpha=1.5):
    """高提升滤波"""
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    mask = cv2.subtract(image, blurred)
    high_boost = cv2.addWeighted(image, 1.0 + alpha, mask, -alpha, 0)
    return high_boost

def unsharp_mask(image, sigma=1.0, strength=1.5):
    """应用 Unsharp Masking 锐化增强"""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

def Test():
    img = cv2.imread("/Users/hezhenbang/Desktop/UBCO/DL-COSC519/GroupProject/Code/jersey-2023/test/images/0/0_6.jpg")
    sr_img = wavelet_sr(img)
    cv2.imwrite("/Users/hezhenbang/Desktop/UBCO/DL-COSC519/GroupProject/Code/jersey-2023/test/images/0/0_6_wavelet_sr.jpg", sr_img)


if __name__ == '__main__':
    from super_image import EdsrModel, ImageLoader
    from PIL import Image
    import requests

    image = Image.open("/Users/hezhenbang/Desktop/UBCO/DL-COSC519/GroupProject/Code/jersey-2023/test/images/0/0_6.jpg")

    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)
    inputs = ImageLoader.load_image(image)
    preds = model(inputs)

    ImageLoader.save_image(preds, "/Users/hezhenbang/Desktop/UBCO/DL-COSC519/GroupProject/Code/jersey-2023/test/images/0/0_6_edsr.jpg")
    ImageLoader.save_compare(inputs, preds, "/Users/hezhenbang/Desktop/UBCO/DL-COSC519/GroupProject/Code/jersey-2023/test/images/0/0_6_edsr_compare.jpg")
