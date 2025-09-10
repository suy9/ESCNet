import logging
import os
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms
import numpy as np
import random
import cv2
from PIL import Image

def get_image(path, size, color_type):
    if color_type.lower() == "rgb":
        image = cv2.imread(path)
    elif color_type.lower() == "gray":
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        print("Select the color_type to return, either to RGB or gray image.")
        return
    if image is None or image.size == 0:
        print("Error: Image is empty.", path)
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    if color_type.lower() == "rgb":
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGB")
    else:
        image = Image.fromarray(image).convert("L")
    return image


def check_state_dict(state_dict, unwanted_prefix="_orig_mod."):
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    return state_dict


def generate_smoothed_gt(gts):
    epsilon = 0.001
    new_gts = (1 - epsilon) * gts + epsilon / 2
    return new_gts


class Logger:
    def __init__(self, config, path="log.txt"):
        self.logger = logging.getLogger(config.name)
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.stdout_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        self.file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def info(self, txt):
        self.logger.info(txt)

    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, path, filename="latest.pth"):
    torch.save(state, os.path.join(path, filename))


def save_tensor_img(tenor_im, path):
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    im.save(path)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def print_info(data):
    if isinstance(data, list):  # 如果是list
        print("List:")
        for item in data:
            print_info(item)  # 递归调用，处理嵌套的list
    elif isinstance(data, torch.Tensor):  # 如果是tensor
        print(f"Tensor shape: {data.shape}")
    else:
        print("Unsupported type")


def refine_foreground(image, mask, r=90):
    if mask.size != image.size:
        mask = mask.resize(image.size)
    image = np.array(image) / 255.0
    mask = np.array(mask) / 255.0
    estimated_foreground = FB_blur_fusion_foreground_estimator_2(image, mask, r=r)
    image_masked = Image.fromarray((estimated_foreground * 255.0).astype(np.uint8))
    return image_masked


def FB_blur_fusion_foreground_estimator_2(image, alpha, r=90):
    # Thanks to the source: https://github.com/Photoroom/fast-foreground-estimation
    alpha = alpha[:, :, None]
    F, blur_B = FB_blur_fusion_foreground_estimator(image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator(image, F, blur_B, alpha, r=6)[0]


def FB_blur_fusion_foreground_estimator(image, F, B, alpha, r=90):
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]

    blurred_FA = cv2.blur(F * alpha, (r, r))
    blurred_F = blurred_FA / (blurred_alpha + 1e-5)

    blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    F = blurred_F + alpha * (image - alpha * blurred_F - (1 - alpha) * blurred_B)
    F = np.clip(F, 0, 1)
    return F, blurred_B


def preproc(image, label, edge, preproc_methods=["flip"]):
    if "flip" in preproc_methods:
        image, label, edge = cv_random_flip(image, label, edge)
    if "crop" in preproc_methods:
        image, label, edge = random_crop(image, label, edge)
    if "rotate" in preproc_methods:
        image, label, edge = random_rotate(image, label, edge)
    if "enhance" in preproc_methods:
        image = color_enhance(image)
    if "pepper" in preproc_methods:
        image= random_pepper(image)
    return image, label, edge


# def preproc(image, label, preproc_methods=["flip"]):
#     if "flip" in preproc_methods:
#         image, label = cv_random_flip(
#             image,
#             label,
#         )
#     if "crop" in preproc_methods:
#         image, label = random_crop(image, label)
#     if "rotate" in preproc_methods:
#         image, label = random_rotate(image, label)
#     if "enhance" in preproc_methods:
#         image = color_enhance(image)
#     if "pepper" in preproc_methods:
#         image = random_pepper(image)
#     return image, label


def cv_random_flip(img, label, edge):

    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)

    return img, label, edge


def random_crop(image, label, edge):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    border = int(min(image_width, image_height) * 0.1)
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )
    cropped_image = image.crop(random_region)
    cropped_label = label.crop(random_region)
    cropped_edge = edge.crop(random_region)
    return (
        cropped_image,
        cropped_label,
        cropped_edge,
    )  


def random_rotate(image, label, edge, angle=15):
    mode = Image.BICUBIC

    if random.random() > 0.8:
        random_angle = np.random.randint(-angle, angle)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)


    return image, label, edge


def color_enhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    contrast_intensity = random.randint(5, 15) / 10.0
    color_intensity = random.randint(0, 20) / 10.0
    sharp_intensity = random.randint(0, 30) / 10.0
    bright_intensity = random.randint(5, 15) / 10.0

    def enhance_single_image(img):
        if img is None:
            return None

        img = ImageEnhance.Brightness(img).enhance(bright_intensity)
        img = ImageEnhance.Contrast(img).enhance(contrast_intensity)
        img = ImageEnhance.Color(img).enhance(color_intensity)
        img = ImageEnhance.Sharpness(img).enhance(sharp_intensity)
        return img

    enhanced_image = enhance_single_image(image)



    return enhanced_image


def random_pepper(image, N=0.0015):
    def pepper_single_image(img):
        img = np.array(img)
        noise_num = int(N * img.shape[0] * img.shape[1])
        for _ in range(noise_num):
            rand_x = random.randint(0, img.shape[0] - 1)
            rand_y = random.randint(0, img.shape[1] - 1)
            img[rand_x, rand_y] = random.randint(0, 1) * 255
        return Image.fromarray(img)

    peppered_image = pepper_single_image(image)



    return peppered_image
