"""
基于傅里叶变换和逆变换的风格增强和风格迁移
图像 -- 傅里叶变换 -- 幅度（风格信息） + 相位（结构信息） -- 逆傅里叶变换
那么通过交换两张图像的幅度，或在幅度域引入扰动（噪声），以改变图像风格
"""
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

def fft_style_transfer(img_src, img_ref, beta=0.5,mix = 1.0):
    # 转换到频域
    f_src = np.fft.fft2(img_src, axes=(0,1))
    f_ref = np.fft.fft2(img_ref, axes=(0,1))

    # 幅度 & 相位
    amp_src, pha_src = np.abs(f_src), np.angle(f_src)
    amp_ref = np.abs(f_ref)

    # 替换低频部分的幅度谱
    h, w, _ = img_src.shape
    b = int(np.floor(min(h, w) * beta))
    c_h, c_w = h//2, w//2
    # amp_src[c_h-b:c_h+b, c_w-b:c_w+b, :] = amp_ref[c_h-b:c_h+b, c_w-b:c_w+b, :]
    #加权融合替代简单替换
    amp_src[c_h - b:c_h + b, c_w - b:c_w + b, :] = ((1 - mix) * amp_src[c_h - b:c_h + b, c_w - b:c_w + b, :]
                                                    + mix * amp_ref[c_h - b:c_h + b, c_w - b:c_w + b, :])

    # 逆变换回图像
    f_new = amp_src * np.exp(1j * pha_src)
    img_new = np.fft.ifft2(f_new, axes=(0,1)).real
    img_new = np.clip(img_new, 0, 255).astype(np.uint8)

    return img_new

def visualize_fft(img,save = False,save_path = None):
    # 转换为灰度（单通道）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # 幅度谱 (log scale)
    magnitude_spectrum = np.log(1 + np.abs(fshift))

    # 相位谱
    phase_spectrum = np.angle(fshift)

    # 可视化
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(gray, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("Magnitude Spectrum (log)")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(phase_spectrum, cmap='gray')
    plt.title("Phase Spectrum")
    plt.axis('off')
    if save:
        plt.savefig(save_path)
    plt.show()


def random_style_fft(img, beta=0.2, noise_scale=0.3):
    f = np.fft.fft2(img, axes=(0,1))
    amp, pha = np.abs(f), np.angle(f)

    h, w, _ = img.shape
    b = int(min(h,w) * beta)
    c_h, c_w = h//2, w//2

    # 在低频区域加噪声
    noise = 1 + noise_scale * (np.random.rand(2*b, 2*b, 3) - 0.5)
    amp[c_h-b:c_h+b, c_w-b:c_w+b, :] *= noise

    f_new = amp * np.exp(1j * pha)
    img_new = np.fft.ifft2(f_new, axes=(0,1)).real
    return np.clip(img_new, 0, 255).astype(np.uint8)

def fda_transfer_background(content, style, mask,
                            beta=0.08, mix=1.0, style_mask=None, feather=0):
    """
    将 content 中的“背景（mask==255）”通过 Fourier low-freq 替换成 style 风格。
    content, style: HxWx3, uint8 或 float32 (0-255)
    mask: HxW 单通道，label: background=255, disc=128, cup=0
    beta: 低频块半径占比 (0~0.5)，常用 0.05~0.2
    mix: 低频替换强度 (0~1)
    style_mask: 可选 HxW（同上标签），若提供则只用 style 的 background 区域来构造 style FFT
    feather: 边界羽化半径（像素，int）。0 表示不羽化，>0 会做高斯模糊软过渡
    """
    # ensure float32
    src = content.astype(np.float32)
    ref = style.astype(np.float32)
    h, w, _ = src.shape
    assert ref.shape[0] == h and ref.shape[1] == w, "content/style must be same size"

    # 背景二值掩码 (bool)
    bg = (mask == 255)

    # 若提供 style_mask，取 style 的 background 部分作为参考（否则用整张 style）
    if style_mask is not None:
        ref_bg = ref * (style_mask[:, :, None] == 255)
    else:
        ref_bg = ref

    # FFT 并把低频移到中心
    F_src = np.fft.fftshift(np.fft.fft2(src, axes=(0,1)), axes=(0,1))
    F_ref = np.fft.fftshift(np.fft.fft2(ref_bg, axes=(0,1)), axes=(0,1))

    amp_src = np.abs(F_src)
    pha_src = np.angle(F_src)
    amp_ref = np.abs(F_ref)

    # 低频块尺寸
    b = max(1, int(min(h, w) * beta))
    c_h, c_w = h // 2, w // 2
    r1, r2 = c_h - b, c_h + b
    c1, c2 = c_w - b, c_w + b

    # 混合替换低频幅度谱
    amp_src[r1:r2, c1:c2, :] = (1.0 - mix) * amp_src[r1:r2, c1:c2, :] + \
                               mix * amp_ref[r1:r2, c1:c2, :]

    # 逆变换回空间域
    F_new = amp_src * np.exp(1j * pha_src)
    F_new = np.fft.ifftshift(F_new, axes=(0,1))
    img_new = np.fft.ifft2(F_new, axes=(0,1)).real
    img_new = np.clip(img_new, 0, 255).astype(np.uint8)

    # 输出：只把 background 区域从 img_new 拷回，其它保持 content
    if feather and feather > 0:
        # 软边界混合，避免突变（mask 先归一化到 0..1，再模糊）
        m = (bg.astype(np.float32))
        k = feather * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)  # 软边界
        m3 = m[:, :, None]
        out = (m3 * img_new.astype(np.float32) + (1.0 - m3) * src).astype(np.uint8)
    else:
        out = content.copy()
        out[bg] = img_new[bg]

    return out

if __name__ == '__main__':
    #1 FFT风格迁移
    content = np.array(Image.open('V0018.png'))
    style = np.array(Image.open('gdrishtiGS_004.png'))
    new_image = fft_style_transfer(content, style,mix = np.random.uniform(0,1))
    new_image = Image.fromarray(new_image)
    new_image.show()
    new_image.save('FFT_result.png')

    #2 可视化分解成的幅度和相位
    visualize_fft(content)
    visualize_fft(style)
    visualize_fft(np.array(new_image))

    #3 Fourier-based 随机风格变换
    noise_scale = np.random.uniform(low=0, high=1)
    random_style_image = random_style_fft(np.array(content),beta=0.5, noise_scale=noise_scale)
    Image.fromarray(random_style_image).show()

    #4 局部傅里叶变换：只对背景做变换
    content_mask = np.array(Image.open('V0018_mask.png').convert('L'))#Image会自动把单通道转化为三通道
    style_mask = np.array(Image.open('gdrishtiGS_004_mask.png').convert('L'))

    fda_background = fda_transfer_background(content, style, content_mask,beta = 0.08,
                                             mix = np.random.uniform(0,1),style_mask = style_mask,feather = 1)
    Image.fromarray(fda_background).show()