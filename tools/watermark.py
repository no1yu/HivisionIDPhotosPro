import math
import os
import textwrap
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont


# 服务启动时先验证思源黑体文件，避免字体上传不完整后等到用户请求时才返回500
ImageFont.truetype(
    os.path.join(os.path.dirname(__file__), "font", "SourceHanSansSC-Regular.otf"),
    size=10,
).getname()


def add_watermark(input_image, text, style, angle=30, color="#171717", opacity=0.30, size=40, space=120):
    """给图片添加重复斜纹水印或居中水印。"""
    origin_image = input_image.convert("RGBA")

    # 水印字号和间距以1000像素短边为基准，防止高清原图缩小预览后看不清
    scale = max(0.5, min(origin_image.size) / 1000)
    watermark_size = max(10, round(size * scale))
    watermark_space = max(10, round(space * scale))
    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "font", "SourceHanSansSC-Regular.otf"), size=watermark_size)

    # 当使用居中水印时，把过长文字按每行八个字符换行
    if style == "central":
        watermark_text = "\n".join(textwrap.wrap(text, width=8))
    else:
        watermark_text = text
    text_lines = watermark_text.splitlines()
    watermark_image = Image.new("RGBA", (max(len(line) for line in text_lines) * watermark_size, round(watermark_size * 1.2 * len(text_lines))))
    ImageDraw.Draw(watermark_image).multiline_text(
        (0, 0),
        watermark_text,
        fill=color,
        font=font,
    )
    difference = ImageChops.difference(watermark_image, Image.new("RGBA", watermark_image.size))
    watermark_image = watermark_image.crop(difference.getbbox())
    alpha = ImageEnhance.Brightness(watermark_image.split()[3]).enhance(opacity)
    watermark_image.putalpha(alpha)
    diagonal = int(math.sqrt(origin_image.width ** 2 + origin_image.height ** 2))
    watermark_mask = Image.new("RGBA", (diagonal, diagonal))

    # 当使用斜纹水印时，把文字铺满整张旋转画布
    if style == "striped":
        y = 0
        row = 0
        while y < diagonal:
            x = -int((watermark_image.width + watermark_space) * 0.5 * row)
            row = (row + 1) % 2
            while x < diagonal:
                watermark_mask.paste(watermark_image, (x, y))
                x += watermark_image.width + watermark_space
            y += watermark_image.height + watermark_space
    else:
        watermark_mask.paste(
            watermark_image,
            (
                int((watermark_mask.width - watermark_image.width) / 2),
                int((watermark_mask.height - watermark_image.height) / 2),
            ),
        )
    watermark_mask = watermark_mask.rotate(angle)

    # 把旋转后的水印裁成原图大小后只合成一次，避免透明度被重复降低
    watermark_mask = watermark_mask.crop(
        (
            int((watermark_mask.width - origin_image.width) / 2),
            int((watermark_mask.height - origin_image.height) / 2),
            int((watermark_mask.width + origin_image.width) / 2),
            int((watermark_mask.height + origin_image.height) / 2),
        )
    )
    return Image.alpha_composite(origin_image, watermark_mask)
