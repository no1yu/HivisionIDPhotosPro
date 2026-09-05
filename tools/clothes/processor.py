import json
from functools import lru_cache
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import onnxruntime

from creator.face.face_detector import detect_faces_yunet
from tools.exceptions import ModelNotFoundError
from tools.model_files import get_model_files


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "tools" / "clothes" / "assets"
ANCHORS_PATH = ASSETS_DIR / "anchors.json"
CLOTHES_COUNTS = {
    1: 17,
    2: 17,
    3: 9,
}
# 换装只能使用YuNet，因为人脸框和关键点坐标的处理是按照YuNet的输出进行量身定制的。
CLOTHES_FACE_DETECTORS = {
    "yunet": detect_faces_yunet,
}
CLOTHES_PARSING_MODELS = {
    "selfieMulticlass",
}
CLOTHES_PARSING_SESSION = None
CLOTHES_PARSING_MODEL_OPTION = None
CLOTHES_PARSING_LOCK = Lock()
MIN_PERSON_SCALE = 0.88
MAX_PERSON_SCALE = 1.15


def _load_clothes_calibrations() -> dict:
    """读取服装自身的领口和肩线结构，不保存任何人物照片参数"""
    try:
        with ANCHORS_PATH.open("r", encoding="utf-8") as file:
            calibrations = json.load(file)
    except (OSError, json.JSONDecodeError):
        raise ValueError("服装结构配置无效")
    if not isinstance(calibrations, dict):
        raise ValueError("服装结构配置无效")
    return calibrations


CLOTHES_CALIBRATIONS = _load_clothes_calibrations()


def _get_clothes_path(category: int, clothes_id: int) -> Path:
    """根据服装分类和编号返回项目内置透明服装模板"""
    if category not in CLOTHES_COUNTS:
        raise ValueError("服装分类无效")

    if clothes_id < 1 or clothes_id > CLOTHES_COUNTS[category]:
        raise ValueError("服装不存在")

    clothes_path = ASSETS_DIR / str(category) / f"{clothes_id:02d}.png"
    if not clothes_path.is_file():
        raise ValueError("服装不存在")
    return clothes_path


def _get_clothes_calibration(category: int, clothes_id: int) -> dict:
    """返回指定服装的一次性模板结构标定"""
    calibration = CLOTHES_CALIBRATIONS.get(f"{category}/{clothes_id:02d}")
    if not isinstance(calibration, dict):
        raise ValueError("服装结构配置不存在")
    shoulders = calibration.get("shoulders")
    if not isinstance(shoulders, list) or len(shoulders) != 2:
        raise ValueError("服装肩线配置无效")
    try:
        normalized_shoulders = np.asarray(shoulders, dtype=np.float32)
    except (TypeError, ValueError):
        raise ValueError("服装肩线配置无效")
    if (
        normalized_shoulders.shape != (2, 2)
        or not np.all(np.isfinite(normalized_shoulders))
        or np.any(normalized_shoulders < 0.0)
        or np.any(normalized_shoulders > 1.0)
        or abs(normalized_shoulders[1, 0] - normalized_shoulders[0, 0]) < 0.05
    ):
        raise ValueError("服装肩线配置无效")
    return calibration


def _alpha_composite(base_image: np.ndarray, overlay_image: np.ndarray) -> np.ndarray:
    """把同尺寸的两个BGRA图层按照Alpha通道合成"""
    base = base_image.astype(np.float32) / 255.0
    overlay = overlay_image.astype(np.float32) / 255.0

    base_alpha = base[:, :, 3:4]
    overlay_alpha = overlay[:, :, 3:4]
    output_alpha = overlay_alpha + base_alpha * (1.0 - overlay_alpha)
    premultiplied_color = (
        overlay[:, :, :3] * overlay_alpha
        + base[:, :, :3] * base_alpha * (1.0 - overlay_alpha)
    )
    output_color = np.divide(
        premultiplied_color,
        output_alpha,
        out=np.zeros_like(premultiplied_color),
        where=output_alpha > 0,
    )
    return np.uint8(
        np.clip(np.concatenate((output_color, output_alpha), axis=2) * 255.0, 0, 255)
    )


def _resize_bgra(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """使用预乘Alpha缩放透明图层，避免透明边缘出现黑边"""
    normalized = image.astype(np.float32) / 255.0
    alpha = normalized[:, :, 3:4]
    premultiplied_color = normalized[:, :, :3] * alpha
    resized_color = cv2.resize(
        premultiplied_color,
        (width, height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    resized_alpha = cv2.resize(
        alpha,
        (width, height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    if resized_alpha.ndim == 2:
        resized_alpha = resized_alpha[:, :, None]

    output_color = np.divide(
        resized_color,
        resized_alpha,
        out=np.zeros_like(resized_color),
        where=resized_alpha > 0,
    )
    return np.uint8(
        np.clip(
            np.concatenate((output_color, resized_alpha), axis=2) * 255.0,
            0,
            255,
        )
    )


def _scale_person_to_target(
    input_image: np.ndarray,
    face: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """根据证件照规格统一双眼距离和眼线位置"""
    image_height, image_width = input_image.shape[:2]
    aspect_ratio = image_height / image_width
    tall_ratio = float(np.clip((aspect_ratio - 1.30) / 0.08, 0.0, 1.0))
    tall_ratio = tall_ratio * tall_ratio * (3.0 - 2.0 * tall_ratio)
    target_eye_distance_ratio = 0.21 + tall_ratio * 0.022

    width_ratio = float(np.clip((image_width - 295) / (413 - 295), 0.0, 1.0))
    target_eye_y_ratio = 0.474 + width_ratio * 0.028

    right_eye_x, right_eye_y = float(face[4]), float(face[5])
    left_eye_x, left_eye_y = float(face[6]), float(face[7])
    eye_center_x = (right_eye_x + left_eye_x) / 2.0
    eye_center_y = (right_eye_y + left_eye_y) / 2.0
    current_eye_distance = abs(left_eye_x - right_eye_x)
    scale = float(
        np.clip(
            image_width * target_eye_distance_ratio / current_eye_distance,
            MIN_PERSON_SCALE,
            MAX_PERSON_SCALE,
        )
    )
    offset_x = image_width / 2.0 - eye_center_x * scale
    offset_y = image_height * target_eye_y_ratio - eye_center_y * scale
    if (
        abs(scale - 1.0) < 0.005
        and abs(offset_x) < 0.5
        and abs(offset_y) < 0.5
    ):
        return input_image, face

    transform = np.array(
        [
            [scale, 0.0, offset_x],
            [0.0, scale, offset_y],
        ],
        dtype=np.float32,
    )

    normalized = input_image.astype(np.float32) / 255.0
    alpha = normalized[:, :, 3:4]
    premultiplied = np.concatenate((normalized[:, :, :3] * alpha, alpha), axis=2)
    scaled = cv2.warpAffine(
        premultiplied,
        transform,
        (image_width, image_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    scaled_alpha = scaled[:, :, 3:4]
    scaled_color = np.divide(
        scaled[:, :, :3],
        scaled_alpha,
        out=np.zeros_like(scaled[:, :, :3]),
        where=scaled_alpha > 0,
    )
    scaled_image = np.uint8(
        np.clip(np.concatenate((scaled_color, scaled_alpha), axis=2) * 255.0, 0, 255)
    )

    scaled_face = face.copy().astype(np.float32)
    scaled_face[0] = scaled_face[0] * scale + offset_x
    scaled_face[1] = scaled_face[1] * scale + offset_y
    scaled_face[2] *= scale
    scaled_face[3] *= scale
    for index in range(4, 14, 2):
        scaled_face[index] = scaled_face[index] * scale + offset_x
        scaled_face[index + 1] = scaled_face[index + 1] * scale + offset_y
    return scaled_image, scaled_face


def _get_parsing_session(parsing_model: str):
    """加载纯CPU服装人体解析模型并复用推理会话"""
    global CLOTHES_PARSING_SESSION, CLOTHES_PARSING_MODEL_OPTION

    with CLOTHES_PARSING_LOCK:
        if (
            CLOTHES_PARSING_SESSION is None
            or CLOTHES_PARSING_MODEL_OPTION != parsing_model
        ):
            model_path, = get_model_files(parsing_model, "selfie_multiclass.onnx")
            session_options = onnxruntime.SessionOptions()
            session_options.log_severity_level = 3
            CLOTHES_PARSING_SESSION = onnxruntime.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            CLOTHES_PARSING_MODEL_OPTION = parsing_model
    return CLOTHES_PARSING_SESSION


def _create_parsing_probabilities(
    input_image: np.ndarray,
    parsing_model: str,
) -> np.ndarray:
    """返回与输入图片同尺寸的六分类人体解析概率"""
    session = _get_parsing_session(parsing_model)
    rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGRA2RGB)
    model_image = cv2.resize(rgb_image, (256, 256), interpolation=cv2.INTER_AREA)
    model_input = model_image.astype(np.float32)[None, :, :, :] / 255.0
    logits = session.run(
        None,
        {session.get_inputs()[0].name: model_input},
    )[0][0]
    logits = logits - np.max(logits, axis=2, keepdims=True)
    probabilities = np.exp(logits)
    probabilities = probabilities / np.sum(probabilities, axis=2, keepdims=True)

    image_height, image_width = input_image.shape[:2]
    return cv2.resize(
        probabilities,
        (image_width, image_height),
        interpolation=cv2.INTER_LINEAR,
    )


def _find_center_span(row_mask: np.ndarray, center_x: int, search_radius: int):
    """返回当前行最靠近人物中心的连续有效区间"""
    image_width = row_mask.shape[0]
    center_x = int(np.clip(center_x, 0, image_width - 1))
    if row_mask[center_x]:
        seed_x = center_x
    else:
        left = max(0, center_x - search_radius)
        right = min(image_width, center_x + search_radius + 1)
        candidates = np.flatnonzero(row_mask[left:right]) + left
        if candidates.size == 0:
            return None
        seed_x = int(candidates[np.argmin(np.abs(candidates - center_x))])

    span_left = seed_x
    span_right = seed_x
    while span_left > 0 and row_mask[span_left - 1]:
        span_left -= 1
    while span_right + 1 < image_width and row_mask[span_right + 1]:
        span_right += 1
    return span_left, span_right


def _measure_neck_geometry(
    input_image: np.ndarray,
    face: np.ndarray,
    parsing_probabilities: np.ndarray,
) -> dict:
    """测量下巴、现有脖子长度以及可安全取样的中央皮肤轮廓"""
    image_height, image_width = input_image.shape[:2]
    face_x, face_y, face_width, face_height = [float(value) for value in face[:4]]
    center_x = face_x + face_width / 2.0
    face_bottom = face_y + face_height
    mouth_y = float((face[11] + face[13]) / 2.0)
    landmark_chin_y = float(
        np.clip(
            mouth_y + face_height * 0.21,
            face_y + face_height * 0.78,
            face_bottom,
        )
    )

    band_left = max(0, round(center_x - face_width * 0.14))
    band_right = min(image_width, round(center_x + face_width * 0.14) + 1)
    face_probability = parsing_probabilities[:, :, 3]
    body_probability = parsing_probabilities[:, :, 2]
    skin_probability = np.clip(face_probability + body_probability, 0.0, 1.0)
    original_alpha = input_image[:, :, 3].astype(np.float32) / 255.0

    transition_start = int(
        np.clip(round(face_y + face_height * 0.75), 0, image_height - 1)
    )
    transition_end = int(
        np.clip(
            round(face_bottom + face_height * 0.08),
            transition_start,
            image_height - 1,
        )
    )
    parsed_chin_y = landmark_chin_y
    transition_rows = 0
    for row in range(transition_start, transition_end + 1):
        row_face = np.mean(face_probability[row, band_left:band_right])
        row_body = np.mean(body_probability[row, band_left:band_right])
        row_alpha = np.mean(original_alpha[row, band_left:band_right])
        if row_face < 0.15 and row_body > 0.60 and row_alpha > 0.50:
            transition_rows += 1
            if transition_rows >= 2:
                parsed_chin_y = float(row - transition_rows + 1)
                break
        else:
            transition_rows = 0

    chin_y = float(
        np.clip(
            max(landmark_chin_y, parsed_chin_y),
            face_y + face_height * 0.78,
            face_bottom + face_height * 0.04,
        )
    )

    scan_start = int(
        np.clip(round(chin_y - face_height * 0.05), 0, image_height - 1)
    )
    scan_end = int(
        np.clip(
            round(chin_y + face_height * 0.26),
            scan_start,
            image_height - 1,
        )
    )
    center_column = round(center_x)
    search_radius = max(2, round(face_width * 0.12))
    minimum_span = max(2, round(face_width * 0.07))
    maximum_half_width = face_width * 0.22
    profile_rows = []
    profile_left = []
    profile_right = []
    skin_bottom = round(chin_y)
    for row in range(scan_start, scan_end + 1):
        valid_skin = (skin_probability[row] > 0.28) & (original_alpha[row] > 0.50)
        span = _find_center_span(valid_skin, center_column, search_radius)
        if span is None or span[1] - span[0] + 1 < minimum_span:
            continue

        span_center = (span[0] + span[1]) / 2.0
        span_half_width = min((span[1] - span[0]) / 2.0, maximum_half_width)
        profile_rows.append(float(row))
        profile_left.append(float(span_center - span_half_width))
        profile_right.append(float(span_center + span_half_width))
        if row >= chin_y:
            skin_bottom = row

    if not profile_rows:
        profile_rows = [float(scan_start), float(scan_end)]
        profile_left = [center_x - face_width * 0.13] * 2
        profile_right = [center_x + face_width * 0.13] * 2

    return {
        "chin_y": chin_y,
        "skin_bottom": float(skin_bottom),
        "skin_probability": skin_probability,
        "profile_rows": np.asarray(profile_rows, dtype=np.float32),
        "profile_left": np.asarray(profile_left, dtype=np.float32),
        "profile_right": np.asarray(profile_right, dtype=np.float32),
    }


def _find_transparent_center_span(alpha_row: np.ndarray, center_x: int):
    """查找服装当前行中与顶部相连的中央透明开口"""
    if alpha_row[center_x] > 32:
        return None

    span_left = center_x
    span_right = center_x
    while span_left > 0 and alpha_row[span_left - 1] <= 32:
        span_left -= 1
    while span_right + 1 < alpha_row.shape[0] and alpha_row[span_right + 1] <= 32:
        span_right += 1
    if span_left == 0 or span_right == alpha_row.shape[0] - 1:
        return None
    return span_left, span_right


def _analyze_clothes(clothes_image: np.ndarray) -> dict:
    """直接从透明素材计算领口中心、宽度和深度，不读取人工锚点"""
    image_height, image_width = clothes_image.shape[:2]
    alpha = clothes_image[:, :, 3]
    _, opaque_x = np.where(alpha > 32)
    if opaque_x.size == 0:
        raise ValueError("服装图片无效")

    center_x = int(np.clip(round((opaque_x.min() + opaque_x.max()) / 2.0), 0, image_width - 1))
    opening_spans = []
    opening_bottom = 0
    closed_rows = 0
    for row in range(min(image_height, max(4, round(image_height * 0.48)))):
        span = _find_transparent_center_span(alpha[row], center_x)
        if span is None:
            if opening_spans:
                closed_rows += 1
                if closed_rows >= 2:
                    break
            continue

        span_width = span[1] - span[0] + 1
        opening_bottom = row
        closed_rows = 0
        if span_width < image_width * 0.08 or span_width > image_width * 0.45:
            continue
        opening_spans.append((row, span[0], span[1]))

    if len(opening_spans) < 3:
        raise ValueError("服装领口识别失败")

    top_spans = opening_spans[:min(4, len(opening_spans))]
    return {
        "center_x": float(center_x),
        "opening_left": float(np.median([span[1] for span in top_spans])),
        "opening_right": float(np.median([span[2] for span in top_spans])),
        "opening_bottom": float(opening_bottom + 1),
    }


@lru_cache(maxsize=sum(CLOTHES_COUNTS.values()))
def _load_clothes_asset(category: int, clothes_id: int) -> tuple[np.ndarray, dict]:
    """读取并分析内置服装素材，同一进程内只执行一次"""
    clothes_path = _get_clothes_path(category, clothes_id)
    clothes_image = cv2.imread(str(clothes_path), cv2.IMREAD_UNCHANGED)
    if clothes_image is None or clothes_image.ndim != 3 or clothes_image.shape[2] != 4:
        raise ValueError("服装图片无效")
    return clothes_image, _analyze_clothes(clothes_image)


def _upscale_clothes(clothes_image: np.ndarray, target_width: int) -> np.ndarray:
    """放大低分辨率服装并轻度恢复衣领边缘，不生成新的衣料纹理"""
    upscale_width = max(clothes_image.shape[1], target_width * 2)
    if upscale_width == clothes_image.shape[1]:
        return clothes_image

    upscale_height = round(
        clothes_image.shape[0] * upscale_width / clothes_image.shape[1]
    )
    normalized = clothes_image.astype(np.float32) / 255.0
    alpha = normalized[:, :, 3:4]
    premultiplied_color = normalized[:, :, :3] * alpha
    resized_color = cv2.resize(
        premultiplied_color,
        (upscale_width, upscale_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    resized_alpha = cv2.resize(
        alpha,
        (upscale_width, upscale_height),
        interpolation=cv2.INTER_LANCZOS4,
    )[:, :, None]
    blurred_color = cv2.GaussianBlur(resized_color, (0, 0), 0.7)
    resized_color = np.clip(resized_color * 1.35 - blurred_color * 0.35, 0.0, 1.0)
    straight_color = np.divide(
        resized_color,
        resized_alpha,
        out=np.zeros_like(resized_color),
        where=resized_alpha > 1e-4,
    )
    return np.uint8(
        np.clip(
            np.concatenate((straight_color, resized_alpha), axis=2) * 255.0,
            0,
            255,
        )
    )


def _create_face_protection_mask(
    input_image: np.ndarray,
    face: np.ndarray,
    parsing_probabilities: np.ndarray,
    chin_y: float,
) -> np.ndarray:
    """生成脸部保护蒙版，衣领不能侵入真实下巴和脸部皮肤"""
    image_height, image_width = input_image.shape[:2]
    face_x, face_y, face_width, face_height = [float(value) for value in face[:4]]
    face_probability = parsing_probabilities[:, :, 3]
    mask = np.clip((face_probability - 0.08) / 0.72, 0.0, 1.0)

    geometry_mask = np.zeros((image_height, image_width), dtype=np.float32)
    geometry_left = int(np.clip(face_x - face_width * 0.08, 0, image_width))
    geometry_right = int(np.clip(face_x + face_width * 1.08, 0, image_width))
    geometry_top = int(np.clip(face_y - face_height * 0.05, 0, image_height))
    geometry_bottom = int(np.clip(chin_y + 2, 0, image_height))
    geometry_mask[
        geometry_top:geometry_bottom + 1,
        geometry_left:geometry_right,
    ] = 1.0
    mask *= geometry_mask
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8))
    return cv2.GaussianBlur(mask, (0, 0), 0.45)


def _fit_clothes_collar(
    clothes_layer: np.ndarray,
    input_image: np.ndarray,
    face: np.ndarray,
    neck_geometry: dict,
    garment_geometry: dict,
) -> np.ndarray:
    """让领口两侧贴合真实脖子，并在领口底部恢复服装原始形状"""
    image_height, image_width = input_image.shape[:2]
    face_x, _, face_width, face_height = [float(value) for value in face[:4]]
    center_x = face_x + face_width / 2.0
    center_column = round(center_x)
    collar_y = garment_geometry["collar_y"]
    collar_spans = []
    skin_spans = []
    scan_top = max(0, round(neck_geometry["chin_y"] - face_height * 0.35))
    scan_bottom = min(image_height, round(collar_y))
    for row in range(scan_top, scan_bottom):
        collar_span = _find_transparent_center_span(
            clothes_layer[row, :, 3],
            center_column,
        )
        if collar_span is None or collar_span[1] - collar_span[0] < face_width * 0.20:
            continue
        skin_span = _find_center_span(
            (neck_geometry["skin_probability"][row] > 0.72)
            & (input_image[row, :, 3] > 240),
            center_column,
            max(2, round(face_width * 0.10)),
        )
        if skin_span is None:
            continue
        collar_spans.append((row, collar_span[0], collar_span[1]))
        skin_spans.append(skin_span)
        if len(collar_spans) >= max(3, round(face_width * 0.018)):
            break

    # 领口高度没有可靠的真实皮肤时保留服装形状，由脖子补全层衔接
    if not collar_spans:
        return clothes_layer
    source_left = float(np.median([span[1] for span in collar_spans]))
    source_right = float(np.median([span[2] for span in collar_spans]))
    target_left = float(np.median([span[0] for span in skin_spans]))
    target_right = float(np.median([span[1] for span in skin_spans]))
    if not target_left < center_x < target_right:
        return clothes_layer

    outer_left = min(source_left, target_left) - face_width * 0.25
    outer_right = max(source_right, target_right) + face_width * 0.25
    columns = np.arange(image_width, dtype=np.float32)
    horizontal_offset = np.interp(
        columns,
        [outer_left, target_left, center_x, target_right, outer_right],
        [outer_left, source_left, center_x, source_right, outer_right],
    ) - columns
    horizontal_offset[(columns <= outer_left) | (columns >= outer_right)] = 0.0
    rows = np.arange(image_height, dtype=np.float32)
    collar_top = collar_spans[0][0]
    progress = np.clip((rows - collar_top) / (collar_y - collar_top), 0.0, 1.0)
    weight = 1.0 - progress * progress * (3.0 - 2.0 * progress)
    map_x = np.float32(columns[None, :] + weight[:, None] * horizontal_offset[None, :])
    map_y = np.broadcast_to(rows[:, None], (image_height, image_width)).copy()

    normalized = clothes_layer.astype(np.float32) / 255.0
    normalized[:, :, :3] *= normalized[:, :, 3:4]
    remapped = cv2.remap(
        normalized,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    remapped[:, :, :3] = np.divide(
        remapped[:, :, :3],
        remapped[:, :, 3:4],
        out=np.zeros_like(remapped[:, :, :3]),
        where=remapped[:, :, 3:4] > 0,
    )
    fitted_layer = clothes_layer.copy()
    changed = (weight[:, None] > 0) & (horizontal_offset[None, :] != 0)
    fitted_layer[changed] = np.uint8(np.clip(remapped[changed] * 255.0, 0, 255))
    garment_geometry["opening_left_x"] = target_left
    garment_geometry["opening_right_x"] = target_right
    return fitted_layer


def _create_clothes_layer(
    input_image: np.ndarray,
    clothes_image: np.ndarray,
    clothes_geometry: dict,
    clothes_calibration: dict,
    face: np.ndarray,
    neck_geometry: dict,
) -> tuple[np.ndarray, dict]:
    """按素材肩线等比缩放整件服装，并把领口开口贴合到真实下巴"""
    image_height, image_width = input_image.shape[:2]
    face_x, _, face_width, face_height = [float(value) for value in face[:4]]
    center_x = face_x + face_width / 2.0
    source_width = clothes_image.shape[1]
    source_height = clothes_image.shape[0]

    shoulder_points = clothes_calibration["shoulders"]
    source_shoulder_left_x = min(point[0] for point in shoulder_points) * source_width
    source_shoulder_right_x = max(point[0] for point in shoulder_points) * source_width
    source_shoulder_center_x = (
        source_shoulder_left_x + source_shoulder_right_x
    ) / 2.0
    source_shoulder_span = max(
        source_shoulder_right_x - source_shoulder_left_x,
        source_width * 0.55,
    )
    source_scale = image_width / source_shoulder_span
    clothes_width = max(1, round(source_width * source_scale))
    clothes_height = max(1, round(source_height * source_scale))
    clothes_image = _upscale_clothes(clothes_image, clothes_width)
    resized_clothes = _resize_bgra(clothes_image, clothes_width, clothes_height)
    source_scale = clothes_width / source_width
    clothes_left = round(center_x - source_shoulder_center_x * source_scale)
    opening_depth_ratio = (
        clothes_geometry["opening_bottom"] * source_scale / face_height
    )
    visible_neck_ratio = 0.14 + float(
        np.clip((opening_depth_ratio - 0.20) * 0.45, 0.0, 0.04)
    )
    visible_neck_length = face_height * visible_neck_ratio
    collar_y = neck_geometry["chin_y"] + visible_neck_length
    best_top = round(
        collar_y - clothes_geometry["opening_bottom"] * source_scale
    )

    clothes_height = resized_clothes.shape[0]
    source_left = max(0, -clothes_left)
    source_top = max(0, -best_top)
    target_left = max(0, clothes_left)
    target_top = max(0, best_top)
    visible_width = min(clothes_width - source_left, image_width - target_left)
    visible_height = min(clothes_height - source_top, image_height - target_top)
    clothes_layer = np.zeros_like(input_image)
    if visible_width > 0 and visible_height > 0:
        clothes_layer[
            target_top:target_top + visible_height,
            target_left:target_left + visible_width,
        ] = resized_clothes[
            source_top:source_top + visible_height,
            source_left:source_left + visible_width,
        ]

    opening_left_x = clothes_left + clothes_geometry["opening_left"] * source_scale
    opening_right_x = clothes_left + clothes_geometry["opening_right"] * source_scale
    garment_geometry = {
        "opening_left_x": opening_left_x,
        "opening_right_x": opening_right_x,
        "collar_y": collar_y,
    }
    clothes_layer = _fit_clothes_collar(
        clothes_layer,
        input_image,
        face,
        neck_geometry,
        garment_geometry,
    )
    return clothes_layer, garment_geometry


def _create_head_layer(
    input_image: np.ndarray,
    face: np.ndarray,
    parsing_probabilities: np.ndarray,
    chin_y: float,
) -> np.ndarray:
    """保留原始头部透明度，供服装图层沿自身轮廓遮挡长发"""
    image_height = input_image.shape[0]
    _, _, _, face_height = [float(value) for value in face[:4]]
    rows = np.arange(image_height, dtype=np.float32)
    face_fade_start = chin_y - face_height * 0.01
    face_fade_end = chin_y + face_height * 0.025
    face_keep = 1.0 - np.clip(
        (rows - face_fade_start) / max(face_fade_end - face_fade_start, 1.0),
        0.0,
        1.0,
    )
    face_keep = face_keep * face_keep * (3.0 - 2.0 * face_keep)
    head_probability = (
        parsing_probabilities[:, :, 1]
        + parsing_probabilities[:, :, 3] * face_keep[:, None]
        + parsing_probabilities[:, :, 5]
    )
    # 原始Alpha已经区分人物与背景，头部保留只比较人物内部类别
    head_probability /= np.sum(parsing_probabilities[:, :, 1:], axis=2)
    mask = np.clip((head_probability - 0.08) / 0.72, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 0.6)

    head_layer = input_image.copy()
    original_alpha = input_image[:, :, 3].astype(np.float32) / 255.0
    head_layer[:, :, 3] = np.uint8(np.clip(original_alpha * mask * 255.0, 0, 255))
    return head_layer


def _create_face_protection_layer(
    input_image: np.ndarray,
    face: np.ndarray,
    parsing_probabilities: np.ndarray,
    chin_y: float,
) -> np.ndarray:
    """在最终结果上恢复被软边误盖的真实脸部像素"""
    mask = _create_face_protection_mask(
        input_image,
        face,
        parsing_probabilities,
        chin_y,
    )
    image_height, image_width = input_image.shape[:2]
    face_x, face_y, face_width, face_height = [float(value) for value in face[:4]]
    center_x = face_x + face_width / 2.0
    jaw_mask = np.zeros((image_height, image_width), dtype=np.float32)
    jaw_polygon = np.array(
        [
            [center_x - face_width * 0.48, face_y + face_height * 0.40],
            [center_x + face_width * 0.48, face_y + face_height * 0.40],
            [center_x + face_width * 0.44, face_y + face_height * 0.58],
            [center_x + face_width * 0.34, face_y + face_height * 0.76],
            [center_x + face_width * 0.18, chin_y - face_height * 0.03],
            [center_x, chin_y + 2],
            [center_x - face_width * 0.18, chin_y - face_height * 0.03],
            [center_x - face_width * 0.34, face_y + face_height * 0.76],
            [center_x - face_width * 0.44, face_y + face_height * 0.58],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(jaw_mask, jaw_polygon, 1.0)
    jaw_mask = cv2.GaussianBlur(jaw_mask, (0, 0), 0.55)
    mask = np.maximum(mask, jaw_mask)
    face_layer = input_image.copy()
    original_alpha = input_image[:, :, 3].astype(np.float32) / 255.0
    face_layer[:, :, 3] = np.uint8(
        np.clip(original_alpha * mask * 255.0, 0, 255)
    )
    return face_layer


def _sample_skin_color(
    input_image: np.ndarray,
    skin_probability: np.ndarray,
    center_x: float,
    face_width: float,
    top: int,
    bottom: int,
) -> np.ndarray:
    """从人物中央真实皮肤计算缺失像素的同源颜色"""
    image_height, image_width = input_image.shape[:2]
    left = int(np.clip(round(center_x - face_width * 0.14), 0, image_width))
    right = int(np.clip(round(center_x + face_width * 0.14), left + 1, image_width))
    top = int(np.clip(top, 0, image_height - 1))
    bottom = int(np.clip(bottom, top + 1, image_height))
    probability_crop = skin_probability[top:bottom, left:right]
    alpha_crop = input_image[top:bottom, left:right, 3]
    valid = (probability_crop > 0.25) & (alpha_crop > 127)
    colors = input_image[top:bottom, left:right, :3][valid]
    if colors.size == 0:
        colors = input_image[top:bottom, left:right, :3].reshape(-1, 3)
    return np.median(colors, axis=0).astype(np.float32)


def _create_neck_layer(
    input_image: np.ndarray,
    face: np.ndarray,
    neck_geometry: dict,
    garment_geometry: dict,
) -> np.ndarray:
    """优先保留真实脖子，原图皮肤不足时再补足衣领所需区域"""
    image_height, image_width = input_image.shape[:2]
    face_x, face_y, face_width, face_height = [float(value) for value in face[:4]]
    center_x = face_x + face_width / 2.0
    chin_y = neck_geometry["chin_y"]
    collar_y = garment_geometry["collar_y"]

    skin_mask = np.clip(
        (neck_geometry["skin_probability"] - 0.16) / 0.56,
        0.0,
        1.0,
    )
    original_alpha = input_image[:, :, 3].astype(np.float32) / 255.0
    real_neck_layer = input_image.copy()
    real_neck_mask = np.zeros((image_height, image_width), dtype=np.float32)
    real_top = max(0, round(face_y + face_height * 0.5))
    real_left = max(0, round(center_x - face_width * 0.4))
    real_right = min(image_width, round(center_x + face_width * 0.4) + 1)
    collar_bottom = min(image_height - 1, round(collar_y + face_height * 0.015))
    real_bottom = min(
        image_height,
        round(max(collar_bottom, neck_geometry["skin_bottom"])) + 1,
    )
    real_neck_mask[real_top:real_bottom, real_left:real_right] = (
        skin_mask[real_top:real_bottom, real_left:real_right]
        * original_alpha[real_top:real_bottom, real_left:real_right]
    )
    real_neck_layer[:, :, 3] = np.uint8(
        np.clip(real_neck_mask * 255.0, 0, 255)
    )
    # 真实皮肤足以覆盖领口时直接保留，不重新拉伸或裁切脖子
    if neck_geometry["skin_bottom"] >= collar_bottom:
        return real_neck_layer

    geometry_mask = np.zeros((image_height, image_width), dtype=np.float32)
    opening_half_width = (
        garment_geometry["opening_right_x"] - garment_geometry["opening_left_x"]
    ) / 2.0
    top_half_width = float(
        np.clip(
            max(opening_half_width * 0.98, face_width * 0.29),
            face_width * 0.29,
            face_width * 0.34,
        )
    )
    bottom_half_width = float(
        np.clip(
            max(opening_half_width * 1.02, top_half_width * 0.96),
            top_half_width * 0.96,
            face_width * 0.35,
        )
    )
    jaw_connection_half_width = float(
        np.clip(
            max(top_half_width * 1.06, face_width * 0.32),
            face_width * 0.32,
            face_width * 0.36,
        )
    )
    neck_top_rise = face_height * 0.16
    output_top = int(np.clip(round(chin_y - neck_top_rise), 0, image_height - 1))
    output_bottom = int(
        np.clip(
            collar_bottom,
            output_top + 1,
            image_height - 1,
        )
    )
    row_count = output_bottom - output_top + 1
    progress = np.linspace(0.0, 1.0, row_count, dtype=np.float32)
    smooth_progress = progress * progress * (3.0 - 2.0 * progress)
    target_half_widths = jaw_connection_half_width + (
        bottom_half_width - jaw_connection_half_width
    ) * smooth_progress
    columns = np.arange(image_width, dtype=np.float32)
    top_curve = chin_y - neck_top_rise * np.power(
        np.clip(
            np.abs(columns - center_x) / max(jaw_connection_half_width, 1.0),
            0.0,
            1.0,
        ),
        1.55,
    )
    for index, row in enumerate(range(output_top, output_bottom + 1)):
        row_half_width = target_half_widths[index]
        inside = (
            (columns >= center_x - row_half_width)
            & (columns <= center_x + row_half_width)
            & (row >= top_curve)
        )
        geometry_mask[row, inside] = 1.0

    connection_padding = max(2, round(face_width * 0.022))
    geometry_mask = cv2.dilate(
        geometry_mask,
        np.ones((1, connection_padding * 2 + 1), dtype=np.uint8),
    )
    geometry_mask = cv2.GaussianBlur(geometry_mask, (0, 0), 0.65)

    available_neck_length = max(neck_geometry["skin_bottom"] - chin_y, 0.0)
    has_neck_texture = available_neck_length >= face_height * 0.035
    if has_neck_texture:
        texture_top = chin_y + max(1.0, face_height * 0.008)
        texture_bottom = min(
            neck_geometry["skin_bottom"],
            chin_y + face_height * 0.10,
        )
    else:
        texture_top = chin_y - face_height * 0.055
        texture_bottom = chin_y - face_height * 0.085

    source_rows = texture_top + (
        texture_bottom - texture_top
    ) * smooth_progress
    profile_rows = neck_geometry["profile_rows"]
    profile_left = neck_geometry["profile_left"]
    profile_right = neck_geometry["profile_right"]
    source_left = np.interp(source_rows, profile_rows, profile_left)
    source_right = np.interp(source_rows, profile_rows, profile_right)
    source_center = float(
        np.clip(
            np.median((source_left + source_right) / 2.0),
            center_x - face_width * 0.035,
            center_x + face_width * 0.035,
        )
    )
    source_half_width = float(
        np.clip(
            np.median((source_right - source_left) / 2.0),
            face_width * 0.08,
            face_width * 0.22,
        )
    )
    source_centers = np.full(
        row_count,
        source_center,
        dtype=np.float32,
    )
    source_half_widths = np.full(
        row_count,
        source_half_width,
        dtype=np.float32,
    )
    map_x = np.zeros((image_height, image_width), dtype=np.float32)
    map_y = np.zeros((image_height, image_width), dtype=np.float32)
    for index, row in enumerate(range(output_top, output_bottom + 1)):
        normalized_x = (columns - center_x) / max(
            target_half_widths[index],
            1.0,
        )
        map_x[row] = source_centers[index] + normalized_x * source_half_widths[index]
        map_y[row] = source_rows[index]

    remapped = cv2.remap(
        input_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    remapped_skin = cv2.remap(
        neck_geometry["skin_probability"],
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    real_mask = skin_mask * geometry_mask * original_alpha
    if not has_neck_texture:
        rows = np.arange(image_height, dtype=np.float32)
        fragment_fade = 1.0 - np.clip(
            (
                rows - (chin_y - face_height * 0.01)
            ) / max(face_height * 0.03, 1.0),
            0.0,
            1.0,
        )
        fragment_fade = fragment_fade * fragment_fade * (
            3.0 - 2.0 * fragment_fade
        )
        real_mask *= fragment_fade[:, None]

    seam_radius = max(1, round(face_width * 0.04))
    sigma = max(seam_radius * 0.75, 0.8)
    normalized_color = input_image[:, :, :3].astype(np.float32) / 255.0
    spread_alpha = cv2.GaussianBlur(real_mask, (0, 0), sigma)
    spread_color = cv2.GaussianBlur(
        normalized_color * real_mask[:, :, None],
        (0, 0),
        sigma,
    )
    spread_color = np.divide(
        spread_color,
        spread_alpha[:, :, None],
        out=np.zeros_like(spread_color),
        where=spread_alpha[:, :, None] > 1e-4,
    )
    remapped_color = remapped[:, :, :3].astype(np.float32) / 255.0
    remapped_valid = (
        (remapped_skin > 0.16)
        & (remapped[:, :, 3] > 64)
    )
    reconstructed_color = np.where(
        remapped_valid[:, :, None],
        remapped_color,
        spread_color,
    )
    fallback_color = _sample_skin_color(
        input_image,
        neck_geometry["skin_probability"],
        center_x,
        face_width,
        round(chin_y - face_height * 0.07),
        round(chin_y),
    ) / 255.0
    missing_color = (~remapped_valid) & (spread_alpha < 1e-4)
    reconstructed_color[missing_color] = fallback_color
    if not has_neck_texture:
        boundary_y = max(chin_y, neck_geometry["skin_bottom"])
        top_color = _sample_skin_color(
            input_image,
            neck_geometry["skin_probability"],
            center_x,
            face_width,
            round(boundary_y - face_height * 0.025),
            round(boundary_y),
        ) / 255.0
        bottom_color = _sample_skin_color(
            input_image,
            neck_geometry["skin_probability"],
            center_x,
            face_width,
            round(chin_y - face_height * 0.065),
            round(chin_y - face_height * 0.035),
        ) / 255.0
        for index, row in enumerate(range(output_top, output_bottom + 1)):
            row_color = top_color + (
                bottom_color - top_color
            ) * smooth_progress[index]
            reconstructed_color[row, geometry_mask[row] > 0.01] = row_color

    real_color = np.where(
        real_mask[:, :, None] > 0.12,
        normalized_color,
        spread_color,
    )
    real_weight = np.clip((spread_alpha - 0.06) / 0.54, 0.0, 1.0)
    output_color = (
        real_color * real_weight[:, :, None]
        + reconstructed_color * (1.0 - real_weight[:, :, None])
    )
    neck_layer = np.zeros_like(input_image)
    neck_layer[:, :, :3] = np.uint8(np.clip(output_color * 255.0, 0, 255))
    neck_layer[:, :, 3] = np.uint8(np.clip(geometry_mask * 255.0, 0, 255))
    # 重建层只补缺失区域，已有真实皮肤保持原样
    return _alpha_composite(neck_layer, real_neck_layer)


def change_clothes(
    input_image: np.ndarray,
    category: int,
    clothes_id: int,
    face_detect_model: str,
    parsing_model: str,
) -> np.ndarray:
    """统一人物比例并按真实下巴和服装开口完成换装"""
    if input_image.ndim != 3 or input_image.shape[2] != 4:
        raise ValueError("换装图片必须是透明证件照")
    face_detector = CLOTHES_FACE_DETECTORS.get(face_detect_model)
    if face_detector is None or parsing_model not in CLOTHES_PARSING_MODELS:
        raise ModelNotFoundError()

    clothes_image, clothes_geometry = _load_clothes_asset(category, clothes_id)
    clothes_calibration = _get_clothes_calibration(category, clothes_id)

    faces = face_detector(input_image[:, :, :3])
    faces_num = 0 if faces is None else len(faces)
    if faces_num != 1:
        raise ValueError("换装图片人脸识别失败")
    input_image, face = _scale_person_to_target(input_image, faces[0])
    parsing_probabilities = _create_parsing_probabilities(input_image, parsing_model)
    neck_geometry = _measure_neck_geometry(
        input_image,
        face,
        parsing_probabilities,
    )
    clothes_layer, garment_geometry = _create_clothes_layer(
        input_image,
        clothes_image,
        clothes_geometry,
        clothes_calibration,
        face,
        neck_geometry,
    )
    neck_layer = _create_neck_layer(
        input_image,
        face,
        neck_geometry,
        garment_geometry,
    )
    head_layer = _create_head_layer(
        input_image,
        face,
        parsing_probabilities,
        neck_geometry["chin_y"],
    )
    face_layer = _create_face_protection_layer(
        input_image,
        face,
        parsing_probabilities,
        neck_geometry["chin_y"],
    )

    # 服装沿自身透明轮廓盖住长发，脸部最后恢复
    dressed_image = _alpha_composite(neck_layer, head_layer)
    dressed_image = _alpha_composite(dressed_image, clothes_layer)
    return _alpha_composite(dressed_image, face_layer)
