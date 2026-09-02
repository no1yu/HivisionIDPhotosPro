from .matting.human_matting import (
    extract_human,
    extract_human_birefnet_lite,
    extract_human_modnet_photographic_portrait_matting,
    extract_human_ppmattingv2,
    extract_human_rmbg,
    extract_human_silueta,
)
from .face.face_detector import detect_face_mtcnn, detect_face_retinaface, detect_face_yunet
from tools.exceptions import ModelNotFoundError


def choose_handler(creator, matting_model_option=None, face_detect_option=None):
    if matting_model_option == "ppMattingV2":
        creator.matting_handler = extract_human_ppmattingv2
    elif matting_model_option == "hivisionModnet":
        creator.matting_handler = extract_human
    elif matting_model_option == "modnetPhotographic":
        creator.matting_handler = extract_human_modnet_photographic_portrait_matting
    elif matting_model_option == "rmbg":
        creator.matting_handler = extract_human_rmbg
    # 当选择Silueta通用抠图模型时，使用Silueta处理器
    elif matting_model_option == "silueta":
        creator.matting_handler = extract_human_silueta
    elif matting_model_option == "birefnet":
        creator.matting_handler = extract_human_birefnet_lite
    else:
        raise ModelNotFoundError()

    if face_detect_option == "yunet":
        creator.detection_handler = detect_face_yunet
    elif face_detect_option == "retinaface":
        creator.detection_handler = detect_face_retinaface
    elif face_detect_option == "mtcnn":
        creator.detection_handler = detect_face_mtcnn
    elif face_detect_option is not None:
        raise ModelNotFoundError()


def choose_colourize_model(model_option):
    if model_option == "ddcolor":
        return "ddcolor"
    raise ModelNotFoundError()


def choose_cartoon_model(model_option):
    if model_option == "cartoon":
        return "cartoon"
    raise ModelNotFoundError()


def choose_deblur_model(model_option):
    if model_option == "realEsrgan":
        return "realEsrgan"
    raise ModelNotFoundError()
