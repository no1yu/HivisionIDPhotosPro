import cv2
import numpy as np

from creator import IDCreator
from creator.choose_handler import choose_handler
from creator.context import Context, Params
from creator.face.face_detector import detect_faces
from tools.exceptions import FaceError
from tools.image_utils import add_background, hex_to_rgb


COUPLE_RED_COLORS = {
    "#5C1117",
    "#D9001B",
    "#D12C25",
    "#FF2121",
    "#FF0000",
    "#FF4C00",
    "#9D2933",
    "#ED333D",
}

WINE_GRADIENT_COLOR = "#5C1117"
WINE_GRADIENT_CENTER = np.array([29, 23, 110], dtype=np.float32)
WINE_GRADIENT_EDGE = np.array([21, 14, 81], dtype=np.float32)


def _detect_couple_faces(image, model):
    """在受控尺寸内检测人脸，再把坐标还原到原图"""
    height, width = image.shape[:2]
    scale = min(1.0, 2000.0 / max(height, width))
    detect_image = image
    if scale < 1.0:
        detect_image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    faces = detect_faces(detect_image, model)
    minimum_face_size = max(24.0, min(detect_image.shape[:2]) * 0.015)
    faces = [
        face
        for face in faces
        if face["rectangle"][2] >= minimum_face_size
        and face["rectangle"][3] >= minimum_face_size
    ]
    if scale < 1.0:
        for face in faces:
            face["rectangle"] = tuple(value / scale for value in face["rectangle"])
            face["landmarks"] = face["landmarks"] / scale
    return faces


def _matting(image, model):
    """按照后台选择的人像分割模型生成原尺寸透明人像"""
    creator = IDCreator()
    choose_handler(creator, model, None)
    ctx = Context(Params())
    ctx.origin_image = image.copy()
    ctx.processing_image = image.copy()
    creator.matting_handler(ctx)
    return ctx.matting_image


def _add_wine_gradient(matting_image):
    """生成人物后方稍亮、四周逐渐压暗的酒红背景"""
    height, width = matting_image.shape[:2]
    x = np.arange(width, dtype=np.float32)[None, :]
    y = np.arange(height, dtype=np.float32)[:, None]
    center_x = width * 0.5
    center_y = height * 0.42
    distance = np.sqrt(
        ((x - center_x) / (width * 0.7)) ** 2
        + ((y - center_y) / (height * 0.8)) ** 2
    )
    distance = np.clip(distance, 0.0, 1.0)
    distance = distance * distance * (3.0 - 2.0 * distance)
    background = (
        WINE_GRADIENT_CENTER[None, None, :] * (1.0 - distance[:, :, None])
        + WINE_GRADIENT_EDGE[None, None, :] * distance[:, :, None]
    )
    alpha = matting_image[:, :, 3:4].astype(np.float32) / 255.0
    return np.clip(
        matting_image[:, :, :3].astype(np.float32) * alpha
        + background * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)


def create_couple_red_photo(image, color, matting_model, face_model):
    """检测两张人脸并合成用户选择的红色背景"""
    normalized_color = color.upper()
    if normalized_color not in COUPLE_RED_COLORS:
        raise ValueError("背景颜色无效")

    faces = _detect_couple_faces(image, face_model)
    if len(faces) != 2:
        raise FaceError("Expected 2 faces, but got {}".format(len(faces)), len(faces))

    matting_image = _matting(image, matting_model)
    if normalized_color == WINE_GRADIENT_COLOR:
        return _add_wine_gradient(matting_image)
    red, green, blue = hex_to_rgb(normalized_color)
    return add_background(
        matting_image,
        bgr=(blue, green, red),
        mode="pure_color",
    ).astype(np.uint8)
