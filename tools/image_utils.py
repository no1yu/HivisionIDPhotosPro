#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PIL import Image
import io
import numpy as np
import cv2


def save_image_dpi_to_bytes(image: np.ndarray, dpi: int = 300):
    """
    设置图像的DPI（每英寸点数）并返回字节流

    :param image: numpy.ndarray, 输入的图像数组
    :param dpi: int, 要设置的DPI值，默认为300
    """
    image = Image.fromarray(image)
    # 创建一个字节流对象
    byte_stream = io.BytesIO()
    # 将图像保存到字节流
    image.save(byte_stream, format="PNG", dpi=(dpi, dpi))
    # 获取字节流的内容
    image_bytes = byte_stream.getvalue()

    return image_bytes


def resize_image_to_kb(input_image, target_size_kb: int, dpi: int = 300):
    """把图片编码成JPEG并压缩到指定KB，不足目标大小时在文件末尾补零"""
    if isinstance(input_image, np.ndarray):
        image = Image.fromarray(input_image)
    else:
        image = input_image

    if image.mode != "RGB":
        image = image.convert("RGB")

    quality = 95
    target_size = target_size_kb * 1024
    while True:
        byte_stream = io.BytesIO()
        image.save(
            byte_stream,
            format="JPEG",
            quality=quality,
            dpi=(dpi, dpi),
        )
        image_bytes = byte_stream.getvalue()

        # 图片不大于目标KB时结束压缩
        if len(image_bytes) <= target_size:
            # 图片小于目标KB时，在JPEG文件末尾补零达到指定大小
            if len(image_bytes) < target_size:
                image_bytes += b"\x00" * (target_size - len(image_bytes))
            return image_bytes

        # 最低质量仍超过目标KB时返回当前图片的动态下限
        if quality == 1:
            minimum_kb = int(np.ceil(len(image_bytes) / 1024))
            raise ValueError(f"目标KB过小，当前图片只能最低{minimum_kb}KB")

        quality = quality - 5 if quality > 5 else 1


def convert_image_format_to_bytes(input_image: Image.Image, target_format: str):
    """把图片转换成指定的JPG、PNG或静态GIF"""
    target_format = target_format.lower()
    byte_stream = io.BytesIO()

    # JPG不支持透明通道，使用白色填充透明区域
    if target_format in ["jpg", "jpeg"]:
        if input_image.mode in ["RGBA", "LA"]:
            rgba_image = input_image.convert("RGBA")
            rgb_image = Image.new("RGB", rgba_image.size, "white")
            rgb_image.paste(rgba_image, mask=rgba_image.getchannel("A"))
        else:
            rgb_image = input_image.convert("RGB")
        rgb_image.save(byte_stream, format="JPEG", quality=95)
    elif target_format == "png":
        input_image.save(byte_stream, format="PNG")
    elif target_format == "gif":
        input_image.save(byte_stream, format="GIF")
    else:
        raise ValueError("目标图片格式无效")

    return byte_stream.getvalue()


def hex_to_rgb(value):
    if not isinstance(value, str) or len(value) != 7 or value[0] != "#":
        raise ValueError("颜色格式无效")
    try:
        return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))
    except ValueError:
        raise ValueError("颜色格式无效")


def generate_gradient(start_color, width, height, mode="updown"):
    # 定义背景颜色
    end_color = (255, 255, 255)  # 白色

    # 创建一个空白图像
    r_out = np.zeros((height, width), dtype=int)
    g_out = np.zeros((height, width), dtype=int)
    b_out = np.zeros((height, width), dtype=int)

    if mode == "updown":
        # 生成上下渐变色
        for y in range(height):
            r = int(
                (y / height) * end_color[0] + ((height - y) / height) * start_color[0]
            )
            g = int(
                (y / height) * end_color[1] + ((height - y) / height) * start_color[1]
            )
            b = int(
                (y / height) * end_color[2] + ((height - y) / height) * start_color[2]
            )
            r_out[y, :] = r
            g_out[y, :] = g
            b_out[y, :] = b

    else:
        # 生成中心渐变色
        img = np.zeros((height, width, 3))
        # 定义椭圆中心和半径
        center = (width // 2, height // 2)
        end_axies = max(height, width)
        # 定义渐变色
        end_color = (255, 255, 255)
        # 绘制椭圆
        for y in range(end_axies):
            axes = (end_axies - y, end_axies - y)
            r = int(
                (y / end_axies) * end_color[0]
                + ((end_axies - y) / end_axies) * start_color[0]
            )
            g = int(
                (y / end_axies) * end_color[1]
                + ((end_axies - y) / end_axies) * start_color[1]
            )
            b = int(
                (y / end_axies) * end_color[2]
                + ((end_axies - y) / end_axies) * start_color[2]
            )

            cv2.ellipse(img, center, axes, 0, 0, 360, (b, g, r), -1)
        b_out, g_out, r_out = cv2.split(np.uint64(img))

    return r_out, g_out, b_out


def add_background(input_image, bgr=(0, 0, 0), mode="pure_color"):
    """
    本函数的功能为为透明图像加上背景。
    :param input_image: numpy.array(4 channels), 透明图像
    :param bgr: tuple, 合成纯色底时的 BGR 值
    :param new_background: numpy.array(3 channels)，合成自定义图像底时的背景图
    :return: output: 合成好的输出图像
    """
    height, width = input_image.shape[0], input_image.shape[1]
    try:
        b, g, r, a = cv2.split(input_image)
    except ValueError:
        raise ValueError(
            "The input image must have 4 channels. 输入图像必须有4个通道，即透明图像。"
        )

    a_cal = a / 255
    if mode == "pure_color":
        # 纯色填充
        b2 = np.full([height, width], bgr[0], dtype=int)
        g2 = np.full([height, width], bgr[1], dtype=int)
        r2 = np.full([height, width], bgr[2], dtype=int)
    elif mode == "updown_gradient":
        b2, g2, r2 = generate_gradient(bgr, width, height, mode="updown")
    else:
        b2, g2, r2 = generate_gradient(bgr, width, height, mode="center")

    output = cv2.merge(
        ((b - b2) * a_cal + b2, (g - g2) * a_cal + g2, (r - r2) * a_cal + r2)
    )

    return output

def add_background_with_image(input_image: np.ndarray, background_image: np.ndarray) -> np.ndarray:
    """
    本函数的功能为为透明图像加上背景。
    :param input_image: numpy.array(4 channels), 透明图像
    :param background_image: numpy.array(3 channels), 背景图像
    :return: output: 合成好的输出图像
    """
    height, width = input_image.shape[:2]
    try:
        b, g, r, a = cv2.split(input_image)
    except ValueError:
        raise ValueError(
            "The input image must have 4 channels. 输入图像必须有4个通道，即透明图像。"
        )

    # 确保背景图像与输入图像大小一致
    background_image = cv2.resize(background_image, (width, height), cv2.INTER_AREA)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    b2, g2, r2 = cv2.split(background_image)

    a_cal = a / 255.0

    # 修正混合公式
    output = cv2.merge(
        (b * a_cal + b2 * (1 - a_cal),
         g * a_cal + g2 * (1 - a_cal),
         r * a_cal + r2 * (1 - a_cal))
    )

    return output.astype(np.uint8)

