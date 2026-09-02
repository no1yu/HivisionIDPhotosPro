from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, Form, File, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
import anyio.to_thread
from creator import IDCreator
from tools.exceptions import FaceError, ModelNotFoundError
from tools.model_files import get_model_files
from creator.photo.layout_calculator import generate_layout_array, generate_layout_image
from creator.choose_handler import (
    choose_handler,
    choose_cartoon_model,
    choose_colourize_model,
    choose_deblur_model,
)
from tools.cartoon.processor import cartoonize
from tools.clothes import change_clothes
from tools.couple import create_couple_red_photo
from tools.image_utils import (
    add_background,
    add_background_with_image,
    convert_image_format_to_bytes,
    hex_to_rgb,
    resize_image_to_kb,
    save_image_dpi_to_bytes,
)
from tools.template.template_calculator import generte_template_photo
from tools.watermark import add_watermark
import numpy as np
import cv2
import onnxruntime
import io
import os
import traceback
from PIL import Image


NSFW_SESSION = None
CARTOON_SESSION = None
CARTOON_DETECTOR_SESSION = None
CARTOON_KEYPOINTS_SESSION = None
CARTOON_MODEL_OPTION = None
DDCOLOR_SESSION = None
DDCOLOR_MODEL_OPTION = None
REAL_ESRGAN_SESSION = None
REAL_ESRGAN_MODEL_OPTION = None


@asynccontextmanager
async def lifespan(_app):
    """准备最多1024个按需创建的图片处理线程"""
    anyio.to_thread.current_default_thread_limiter().total_tokens = 1024
    yield


app = FastAPI(title="HivisionIDPhotos Pro", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


# 图片接口失败统一返回404，并返回具体原因
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_request: Request, exception: StarletteHTTPException):
    return JSONResponse(
        status_code=404,
        content={"code": 404, "msg": str(exception.detail)},
    )


# FastAPI参数校验失败时也使用项目自己的错误结构
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request, _exception: RequestValidationError
):
    return JSONResponse(
        status_code=404,
        content={"code": 404, "msg": "图片接口参数错误"},
    )


# 请求使用的模型尚未放入时，统一返回模型缺失提示
@app.exception_handler(ModelNotFoundError)
async def model_not_found_exception_handler(
    _request: Request, exception: ModelNotFoundError
):
    return JSONResponse(
        status_code=404,
        content={"code": 404, "msg": str(exception)},
    )


# 未单独处理的模型或图片异常保留完整日志，并返回具体原因
@app.exception_handler(Exception)
async def image_exception_handler(_request: Request, exception: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=404,
        content={"code": 404, "msg": "图片处理失败：" + str(exception)},
    )


# 访问域名根路径时显示接口服务状态
@app.get("/")
def status():
    return Response(content='''<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>图片处理服务</title>
    <style>
        *{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:#f3f6fb;color:#182230;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif}.card{width:520px;padding:58px 60px;text-align:center;background:#fff;border:1px solid #e6ebf2;border-radius:20px;box-shadow:0 18px 55px rgba(27,43,65,.09)}.icon{width:68px;height:68px;margin:0 auto 25px;display:grid;place-items:center;border-radius:50%;background:#eaf8f0}.check{width:25px;height:14px;border-left:4px solid #25a468;border-bottom:4px solid #25a468;transform:rotate(-45deg) translate(2px,-2px)}h1{margin:0;font-size:27px;font-weight:650;letter-spacing:.2px}
    </style>
</head>
<body>
    <main class="card">
        <div class="icon"><span class="check"></span></div>
        <h1>运行正常</h1>
    </main>
</body>
</html>''', media_type="text/html")


# 功能演示页，删除demo目录后也不影响其它接口运行
demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo")
app.mount(
    "/demo",
    StaticFiles(directory=demo_dir, html=True, check_dir=False),
    name="demo",
)


def load_input_image(input_image: UploadFile, flags: int = cv2.IMREAD_COLOR):
    """读取上传的图片，并确认它是一张可以正常打开的图片"""
    image_bytes = input_image.file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), flags)
    # 当上传内容不能被读取成图片时，直接告诉调用方图片无效
    if image is None:
        raise HTTPException(status_code=404, detail="图片文件无效")
    return image


def parse_hex_color(color: str):
    """只接受带#号的六位RGB十六进制颜色"""
    try:
        return hex_to_rgb(color)
    except ValueError:
        raise HTTPException(status_code=404, detail="颜色格式无效")


def resize_image_to_kb_response(image, kb: int, dpi: int):
    """压缩到用户指定KB并把动态压缩下限转换为业务错误"""
    try:
        return resize_image_to_kb(image, kb, dpi)
    except ValueError as exception:
        raise HTTPException(status_code=404, detail=str(exception))


@app.post("/checkImg")
def check_image_nsfw(
    input_image: UploadFile = File(...),
):
    """使用NSFWJS MobileNetV2检查图片，并返回五项分类分数"""
    global NSFW_SESSION

    # 首次请求会把鉴黄模型加载到内存，执行完不释放，方便下次请求进来处理更快
    if NSFW_SESSION is None:
        nsfw_model_path, = get_model_files("nsfwjs", "nsfwjs.onnx")

        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 3
        NSFW_SESSION = onnxruntime.InferenceSession(
            nsfw_model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    image = load_input_image(input_image, cv2.IMREAD_COLOR)
    model_image = Image.fromarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ).resize((224, 224), Image.Resampling.NEAREST)
    image_tensor = np.ascontiguousarray(
        np.asarray(model_image, dtype=np.float32)[None] / 255.0,
        dtype=np.float32,
    )
    scores = NSFW_SESSION.run(
        ["dense_3"],
        {"input_1": image_tensor},
    )[0][0]

    return {
        "code": 200,
        "msg": "检测成功",
        "data": {
            "drawings": float(scores[0]),
            "hentai": float(scores[1]),
            "neutral": float(scores[2]),
            "porn": float(scores[3]),
            "sexy": float(scores[4]),
        },
    }


def image_response(
    image: np.ndarray,
    dpi: int = 300,
    image_format: str = "png",
    kb: int = None,
):
    """把处理好的图片直接作为图片文件返回"""
    # 指定KB时按照原版逻辑输出JPEG，并在质量压缩后补齐目标字节数
    if kb is not None:
        return Response(
            content=resize_image_to_kb_response(image, kb, dpi),
            media_type="image/jpeg",
        )

    # 当用户要求JPG格式时，把图片转换成JPG
    if (image_format or "png").lower() in ["jpg", "jpeg"]:
        pil_image = Image.fromarray(image)
        # 当图片不是普通三通道图片时，先转换成JPG支持的三通道图片
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        byte_stream = io.BytesIO()
        pil_image.save(byte_stream, format="JPEG", quality=95, dpi=(dpi, dpi))
        image_bytes = byte_stream.getvalue()
        media_type = "image/jpeg"
    else:
        image_bytes = save_image_dpi_to_bytes(image, dpi)
        media_type = "image/png"
    return Response(content=image_bytes, media_type=media_type)


def create_idphoto_result(
    image: np.ndarray,
    height: int,
    width: int,
    human_matting_model: str,
    face_detect_model: str,
    face_align: bool = False,
    head_measure_ratio: float = 0.2,
    head_height_ratio: float = 0.45,
    top_distance_max: float = 0.12,
    top_distance_min: float = 0.10,
    whitening_strength: int = 0,
    brightness_strength: float = 0,
    contrast_strength: float = 0,
    sharpen_strength: float = 0,
    saturation_strength: float = 0,
):
    """执行证件照核心处理，供普通证件照、美式证件照和模板照共同使用"""
    # 每个请求使用独立处理上下文，防止并发请求互相覆盖图片状态
    creator = IDCreator()
    choose_handler(creator, human_matting_model, face_detect_model)
    try:
        return creator(
            image,
            size=(int(height), int(width)),
            head_measure_ratio=head_measure_ratio,
            head_height_ratio=head_height_ratio,
            head_top_range=(top_distance_max, top_distance_min),
            face_alignment=face_align,
            whitening_strength=whitening_strength,
            brightness_strength=brightness_strength,
            contrast_strength=contrast_strength,
            sharpen_strength=sharpen_strength,
            saturation_strength=saturation_strength,
        )
    except FaceError as exception:
        if exception.face_num == 0:
            raise HTTPException(status_code=404, detail="未检测到人脸，制作失败")
        raise HTTPException(status_code=404, detail="检测到多个人脸，请上传单人照片")


@app.post("/idphoto")
def idphoto_inference(
    input_image: UploadFile = File(...),
    height: int = Form(413),
    width: int = Form(295),
    human_matting_model: str = Form("ppMattingV2"),
    face_detect_model: str = Form("yunet"),
    hd: bool = Form(True),
    dpi: int = Form(300),
    face_align: bool = Form(False),
    head_measure_ratio: float = Form(0.2),
    head_height_ratio: float = Form(0.45),
    top_distance_max: float = Form(0.12),
    top_distance_min: float = Form(0.10),
    whitening_strength: int = Form(0),
    brightness_strength: float = Form(0),
    contrast_strength: float = Form(0),
    sharpen_strength: float = Form(0),
    saturation_strength: float = Form(0),
):
    """生成标准或高清透明证件照，hd决定本次直接返回哪一张图片"""
    image = load_input_image(input_image, cv2.IMREAD_COLOR)
    result = create_idphoto_result(
        image,
        height,
        width,
        human_matting_model,
        face_detect_model,
        face_align,
        head_measure_ratio,
        head_height_ratio,
        top_distance_max,
        top_distance_min,
        whitening_strength,
        brightness_strength,
        contrast_strength,
        sharpen_strength,
        saturation_strength,
    )
    output = result.hd if hd else result.standard
    return image_response(cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA), dpi)


@app.post("/human_matting")
def human_matting_inference(
    input_image: UploadFile = File(...),
    human_matting_model: str = Form("ppMattingV2"),
    dpi: int = Form(300),
):
    """抠出透明背景人像并返回PNG二进制"""
    image = load_input_image(input_image, cv2.IMREAD_COLOR)

    # 每个请求使用独立处理上下文，防止并发请求互相覆盖图片状态
    creator = IDCreator()
    choose_handler(creator, human_matting_model, None)
    try:
        result = creator(image, change_bg_only=True)
    except FaceError:
        raise HTTPException(status_code=404, detail="未识别到有效人像，抠图失败")
    return image_response(cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA), dpi)


@app.post("/add_background")
def photo_add_background(
    input_image: UploadFile = File(...),
    color: str = Form("#000000"),
    dpi: int = Form(300),
    render: int = Form(0),
    kb: int = Form(None),
):
    """给透明证件照添加纯色或渐变背景并返回成片"""
    image = load_input_image(input_image, cv2.IMREAD_UNCHANGED)
    # 当换底方式不是纯色、上下渐变或中心渐变时，拒绝处理
    if render not in [0, 1, 2]:
        raise HTTPException(status_code=404, detail="背景渲染类型无效")
    if kb is not None and kb < 1:
        raise HTTPException(status_code=404, detail="目标KB必须大于0")
    rgb = parse_hex_color(color)
    result = add_background(
        image,
        bgr=(rgb[2], rgb[1], rgb[0]),
        mode=["pure_color", "updown_gradient", "center_gradient"][render],
    ).astype(np.uint8)
    return image_response(cv2.cvtColor(result, cv2.COLOR_RGB2BGR), dpi, kb=kb)


@app.post("/change_clothes")
def photo_change_clothes(
    input_image: UploadFile = File(...),
    clothes_category: int = Form(0),
    clothes_id: int = Form(0),
    clothes_face_detect_model: str = Form(...),
    clothes_parsing_model: str = Form(...),
    color: str = Form("#ffffff"),
    dpi: int = Form(300),
    render: int = Form(0),
    kb: int = Form(None),
):
    """给透明证件照更换服装，再按照当前编辑状态合成背景"""
    image = load_input_image(input_image, cv2.IMREAD_UNCHANGED)

    # 背景渲染方式不在约定范围内时，停止处理
    if render not in [0, 1, 2]:
        raise HTTPException(status_code=404, detail="背景渲染类型无效")

    # 目标KB小于1时，停止处理
    if kb is not None and kb < 1:
        raise HTTPException(status_code=404, detail="目标KB必须大于0")

    # 取消换装时分类和服装编号必须同时为0
    if clothes_category == 0 and clothes_id != 0:
        raise HTTPException(status_code=404, detail="取消换装参数无效")

    dressed_image = image

    # 服装分类不是0时，给透明证件照换上用户选择的服装
    if clothes_category != 0:
        try:
            dressed_image = change_clothes(
                image,
                clothes_category,
                clothes_id,
                clothes_face_detect_model,
                clothes_parsing_model,
            )
        # 服装分类、编号或素材不符合要求时，返回具体业务原因
        except ValueError as exception:
            raise HTTPException(status_code=404, detail=str(exception))

    rgb = parse_hex_color(color)
    result = add_background(
        dressed_image,
        bgr=(rgb[2], rgb[1], rgb[0]),
        mode=["pure_color", "updown_gradient", "center_gradient"][render],
    ).astype(np.uint8)
    return image_response(cv2.cvtColor(result, cv2.COLOR_RGB2BGR), dpi, kb=kb)


@app.post("/couple_red_photo")
def couple_red_photo(
    input_image: UploadFile = File(...),
    color: str = Form("#5C1117"),
    human_matting_model: str = Form("ppMattingV2"),
    face_detect_model: str = Form("yunet"),
    dpi: int = Form(300),
):
    """保留原图尺寸，为包含两个人的照片更换红色背景"""
    image = load_input_image(input_image, cv2.IMREAD_COLOR)
    try:
        result = create_couple_red_photo(
            image,
            color,
            human_matting_model,
            face_detect_model,
        )
    except FaceError as exception:
        if exception.face_num == 0:
            raise HTTPException(status_code=404, detail="未检测到人脸，请上传包含两个人的照片")
        if exception.face_num == 1:
            raise HTTPException(status_code=404, detail="只检测到一个人，请上传包含两个人的照片")
        raise HTTPException(status_code=404, detail="检测到超过两个人，请上传只包含两个人的照片")
    except ValueError as exception:
        raise HTTPException(status_code=404, detail=str(exception))
    return image_response(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), dpi)


@app.post("/american_idphoto")
def american_idphoto(
    input_image: UploadFile = File(...),
    height: int = Form(600),
    width: int = Form(600),
    human_matting_model: str = Form("ppMattingV2"),
    face_detect_model: str = Form("yunet"),
    face_align: bool = Form(True),
    background_image: UploadFile = File(None),
    background: str = Form("american"),
    dpi: int = Form(300),
    kb: int = Form(None),
):
    """从原图制作透明证件照并直接合成美式背景"""
    image = load_input_image(input_image, cv2.IMREAD_COLOR)
    if kb is not None and kb < 1:
        raise HTTPException(status_code=404, detail="目标KB必须大于0")
    result = create_idphoto_result(
        image,
        height,
        width,
        human_matting_model,
        face_detect_model,
        face_align,
    )
    # 当请求上传了背景图片时，使用上传的背景图片
    if background_image is not None:
        background_bytes = background_image.file.read()
        background_result = cv2.imdecode(np.frombuffer(background_bytes, np.uint8), cv2.IMREAD_COLOR)
    # 当请求使用美式背景时，读取项目内置的美式背景图片
    elif background == "american":
        background_result = cv2.imread(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "tools",
            "american",
            "assets",
            "american-style.png",
        ))
    else:
        raise HTTPException(status_code=404, detail="背景类型无效")
    # 当背景图片没有读取成功时，停止合成
    if background_result is None:
        raise HTTPException(status_code=404, detail="背景图片无效")
    person_image = cv2.cvtColor(result.standard, cv2.COLOR_BGRA2RGBA)
    return image_response(
        np.uint8(add_background_with_image(person_image, background_result)),
        dpi,
        kb=kb,
    )


@app.post("/generate_layout_photos")
def generate_layout_photos(
    input_image: UploadFile = File(...),
    height: int = Form(413),
    width: int = Form(295),
    dpi: int = Form(300),
    layout_size: str = Form(None),
    layout_height: int = Form(None),
    layout_width: int = Form(None),
    crop_line: bool = Form(False),
    kb: int = Form(None),
):
    """将单张证件照排版到指定相纸画布"""
    image = load_input_image(input_image, cv2.IMREAD_COLOR)
    if kb is not None and kb < 1:
        raise HTTPException(status_code=404, detail="目标KB必须大于0")
    # 画布预设和自定义画布尺寸不能同时传入
    if layout_size is not None and (layout_height is not None or layout_width is not None):
        raise HTTPException(status_code=404, detail="画布预设和自定义画布尺寸不能同时传入")

    # 自定义画布高度和宽度必须同时传入
    if (layout_height is None) != (layout_width is None):
        raise HTTPException(status_code=404, detail="自定义画布高度和宽度必须同时传入")

    # 当用户填写了自定义画布宽高时，使用用户填写的画布尺寸
    if layout_height is not None and layout_width is not None:
        if layout_height < 1 or layout_width < 1:
            raise HTTPException(status_code=404, detail="自定义画布尺寸必须大于0")
        canvas_height = int(layout_height)
        canvas_width = int(layout_width)
    else:
        layout_sizes = {
            "six_inch": (1205, 1795),
            "five_inch": (1051, 1500),
            "a4": (2479, 3508),
            "three_r": (1051, 1500),
            "four_r": (1205, 1795),
        }
        layout_size = layout_size or "six_inch"
        if layout_size.lower() not in layout_sizes:
            raise HTTPException(status_code=404, detail="画布预设无效")
        canvas_height, canvas_width = layout_sizes[layout_size.lower()]
    typography_arr, typography_rotate = generate_layout_array(
        input_height=int(height),
        input_width=int(width),
        LAYOUT_HEIGHT=canvas_height,
        LAYOUT_WIDTH=canvas_width,
    )
    result = generate_layout_image(
        image,
        typography_arr,
        typography_rotate,
        height=int(height),
        width=int(width),
        crop_line=crop_line,
        LAYOUT_HEIGHT=canvas_height,
        LAYOUT_WIDTH=canvas_width,
    ).astype(np.uint8)
    return image_response(
        cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
        dpi,
        kb=kb,
    )


@app.post("/generate_template_photos")
def generate_template_photos(
    input_image: UploadFile = File(...),
    height: int = Form(413),
    width: int = Form(295),
    human_matting_model: str = Form("ppMattingV2"),
    face_detect_model: str = Form("yunet"),
    face_align: bool = Form(True),
    template_name: str = Form("template_1"),
    color: str = Form("#438edb"),
    dpi: int = Form(300),
    kb: int = Form(None),
):
    """从原图制作透明证件照并直接生成社交媒体模板"""
    # 当模板名字不是项目支持的两个模板时，拒绝处理
    if template_name not in ["template_1", "template_2"]:
        raise HTTPException(status_code=404, detail="模板名称无效")
    if kb is not None and kb < 1:
        raise HTTPException(status_code=404, detail="目标KB必须大于0")
    image = load_input_image(input_image, cv2.IMREAD_COLOR)
    result = create_idphoto_result(
        image,
        height,
        width,
        human_matting_model,
        face_detect_model,
        face_align,
    )
    # 给透明证件照填充用户选择的背景颜色，再交给模板功能排版
    rgb = parse_hex_color(color)
    image = add_background(
        result.standard,
        bgr=(rgb[2], rgb[1], rgb[0]),
        mode="pure_color",
    ).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_response(
        generte_template_photo(template_name, image),
        dpi,
        kb=kb,
    )


@app.post("/watermark")
def watermark(
    input_image: UploadFile = File(...),
    text: str = Form(...),
    style: str = Form("striped"),
    angle: int = Form(30),
    color: str = Form("#171717"),
    opacity: float = Form(0.30),
    size: int = Form(40),
    space: int = Form(120),
    dpi: int = Form(300),
    kb: int = Form(None),
):
    """给上传的图片添加文字水印并返回PNG文件"""
    image = load_input_image(input_image, cv2.IMREAD_UNCHANGED)
    if kb is not None and kb < 1:
        raise HTTPException(status_code=404, detail="目标KB必须大于0")
    parse_hex_color(color)

    # 当上传的是灰度图片时，先转成Pillow可以继续处理的RGB图片
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image).convert("RGB")

    # 当上传图片带有透明通道时，把OpenCV颜色顺序转换成Pillow颜色顺序
    elif image.shape[2] == 4:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    else:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result = add_watermark(pil_image, text, style, angle, color, opacity, size, space)
    if kb is not None:
        return Response(
            content=resize_image_to_kb_response(result, kb, dpi),
            media_type="image/jpeg",
        )
    byte_stream = io.BytesIO()
    result.save(byte_stream, format="PNG", dpi=(dpi, dpi))
    return Response(content=byte_stream.getvalue(), media_type="image/png")


@app.post("/set_kb")
def set_kb(
    input_image: UploadFile = File(...),
    dpi: int = Form(300),
    kb: int = Form(50),
):
    """把上传的图片压缩到指定KB并返回JPEG文件"""
    if kb < 1:
        raise HTTPException(status_code=404, detail="目标KB必须大于0")
    image = load_input_image(input_image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Response(
        content=resize_image_to_kb_response(image, kb, dpi),
        media_type="image/jpeg",
    )


@app.post("/convert_image_format")
def convert_image_format(
    input_image: UploadFile = File(...),
    target_format: str = Form(...),
):
    """把上传的图片转换成指定格式"""
    target_format = target_format.lower()
    if target_format not in ["jpg", "jpeg", "png", "gif"]:
        raise HTTPException(status_code=404, detail="目标图片格式无效")

    image = load_input_image(input_image, cv2.IMREAD_UNCHANGED)
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image)
    elif image.shape[2] == 4:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    else:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    media_type = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
    }[target_format]
    return Response(
        content=convert_image_format_to_bytes(pil_image, target_format),
        media_type=media_type,
    )


@app.post("/cartoon")
def cartoon(
    input_image: UploadFile = File(...),
    cartoon_model: str = Form("cartoon"),
):
    """使用DCT-Net把照片转换成立体动漫风"""
    global CARTOON_SESSION, CARTOON_DETECTOR_SESSION, CARTOON_KEYPOINTS_SESSION, CARTOON_MODEL_OPTION

    model_directory = choose_cartoon_model(cartoon_model)

    (
        cartoon_model_path,
        detector_model_path,
        keypoints_model_path,
        alpha_path,
    ) = get_model_files(
        model_directory,
        "cartoon.onnx",
        "detector.onnx",
        "keypoints.onnx",
        "alpha.jpg",
    )

    image = load_input_image(input_image, cv2.IMREAD_COLOR)

    # 首次请求会把动漫照模型加载到内存，执行完不释放，方便下次请求进来处理更快
    if CARTOON_SESSION is None or CARTOON_MODEL_OPTION != cartoon_model:
        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 3
        session_options.enable_cpu_mem_arena = False
        cartoon_session = onnxruntime.InferenceSession(
            cartoon_model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        detector_session = onnxruntime.InferenceSession(
            detector_model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        keypoints_session = onnxruntime.InferenceSession(
            keypoints_model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        CARTOON_SESSION = cartoon_session
        CARTOON_DETECTOR_SESSION = detector_session
        CARTOON_KEYPOINTS_SESSION = keypoints_session
        CARTOON_MODEL_OPTION = cartoon_model

    result = cartoonize(
        image,
        CARTOON_SESSION,
        CARTOON_DETECTOR_SESSION,
        CARTOON_KEYPOINTS_SESSION,
        alpha_path,
    )

    # DCT-Net返回BGR图片，转换成RGB后交给Pillow保存
    return image_response(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


@app.post("/colourizeImg")
def colourize(
    input_image: UploadFile = File(...),
    colourize_model: str = Form("ddcolor"),
    dpi: int = Form(300),
):
    """使用DDColor ModelScope给黑白照片上色，并直接返回原图尺寸的PNG图片"""
    global DDCOLOR_SESSION, DDCOLOR_MODEL_OPTION

    image = load_input_image(input_image, cv2.IMREAD_COLOR)
    original_height, original_width = image.shape[:2]
    image_float = image.astype(np.float32)/255.0
    original_l = cv2.cvtColor(image_float, cv2.COLOR_BGR2Lab)[:, :, :1]
    model_directory = choose_colourize_model(colourize_model)
    model_path, = get_model_files(model_directory, "ddcolor.onnx")

    # 首次请求会把黑白照片上色模型加载到内存，执行完不释放，方便下次请求进来处理更快
    if DDCOLOR_SESSION is None or DDCOLOR_MODEL_OPTION != colourize_model:
        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 3
        session_options.enable_cpu_mem_arena = False
        DDCOLOR_SESSION = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        DDCOLOR_MODEL_OPTION = colourize_model

    # 模型只预测Lab颜色通道，原图亮度和清晰度不经过缩小后的模型
    model_image = cv2.resize(
        image_float,
        (512, 512),
        interpolation=cv2.INTER_LINEAR,
    )
    model_l = cv2.cvtColor(model_image, cv2.COLOR_BGR2Lab)[:, :, :1]
    model_rgb = cv2.cvtColor(
        np.concatenate(
            (model_l, np.zeros_like(model_l), np.zeros_like(model_l)),
            axis=2,
        ),
        cv2.COLOR_Lab2RGB,
    )
    image_tensor = np.ascontiguousarray(
        np.transpose(model_rgb, (2, 0, 1))[None],
        dtype=np.float32,
    )
    result_ab = DDCOLOR_SESSION.run(
        None,
        {DDCOLOR_SESSION.get_inputs()[0].name: image_tensor},
    )[0][0]
    result_ab = np.transpose(result_ab, (1, 2, 0))

    # 把模型生成的颜色恢复到原图大小，再和原图亮度合并，保留原始分辨率与纹理
    result_ab = cv2.resize(
        result_ab,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )
    result = cv2.cvtColor(
        np.concatenate((original_l, result_ab), axis=2),
        cv2.COLOR_Lab2BGR,
    )
    result = np.clip(result*255.0, 0, 255).round().astype(np.uint8)

    # DDColor ModelScope合成结果是BGR，转换成RGB后交给Pillow保存，避免红蓝通道互换
    return image_response(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), dpi)


@app.post("/deblur")
def deblur(
    input_image: UploadFile = File(...),
    deblur_model: str = Form("realEsrgan"),
    dpi: int = Form(300),
):
    """使用Real-ESRGAN增强低清图片，并直接返回处理后的PNG图片"""
    global REAL_ESRGAN_SESSION, REAL_ESRGAN_MODEL_OPTION

    image = load_input_image(input_image, cv2.IMREAD_COLOR)
    original_height, original_width = image.shape[:2]
    model_image = image

    # 当手机原图超过512像素时，先等比例缩小，控制Real-ESRGAN需要处理的分块数量
    if max(original_height, original_width) > 512:
        scale = 512.0 / max(original_height, original_width)
        model_image = cv2.resize(
            image,
            (round(original_width * scale), round(original_height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    model_height, model_width = model_image.shape[:2]
    model_directory = choose_deblur_model(deblur_model)
    model_path, _model_data_path = get_model_files(
        model_directory,
        "real_esrgan_general_x4v3.onnx",
        "real_esrgan_general_x4v3.data",
    )

    # 首次请求会把图片清晰度增强模型加载到内存，执行完不释放，方便下次请求进来处理更快
    if REAL_ESRGAN_SESSION is None or REAL_ESRGAN_MODEL_OPTION != deblur_model:
        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 3
        session_options.enable_cpu_mem_arena = False
        REAL_ESRGAN_SESSION = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        REAL_ESRGAN_MODEL_OPTION = deblur_model

    # 官方量化模型固定接收128×128图片，四周保留8像素内容可以减轻分块接缝
    model_image = cv2.cvtColor(model_image, cv2.COLOR_BGR2RGB)
    core_size = 112
    tile_padding = 8
    output_scale = 4
    padded_height = ((model_height+core_size-1)//core_size)*core_size
    padded_width = ((model_width+core_size-1)//core_size)*core_size
    padded_image = cv2.copyMakeBorder(
        model_image,
        tile_padding,
        tile_padding+padded_height-model_height,
        tile_padding,
        tile_padding+padded_width-model_width,
        cv2.BORDER_REFLECT_101,
    )
    result_canvas = np.empty(
        (padded_height*output_scale, padded_width*output_scale, 3),
        dtype=np.uint8,
    )
    input_name = REAL_ESRGAN_SESSION.get_inputs()[0].name

    # 逐块处理可以兼容任意宽高，同时避免一次创建超大的模型中间特征
    for top in range(0, padded_height, core_size):
        for left in range(0, padded_width, core_size):
            image_tile = padded_image[
                top:top+core_size+tile_padding*2,
                left:left+core_size+tile_padding*2,
            ]
            image_tensor = np.ascontiguousarray(
                np.transpose(image_tile, (2, 0, 1))[None],
                dtype=np.uint8,
            )
            output = REAL_ESRGAN_SESSION.run(
                None,
                {input_name: image_tensor},
            )[0][0]
            output = np.transpose(output, (1, 2, 0))

            # 官方W8A8输出需要按照模型元数据中的零点和比例还原成正常RGB像素
            output = (output.astype(np.float32)-25.0)*0.004972138907760382
            output = np.clip(output*255.0, 0, 255).astype(np.uint8)
            output = output[
                tile_padding*output_scale:(tile_padding+core_size)*output_scale,
                tile_padding*output_scale:(tile_padding+core_size)*output_scale,
            ]
            result_canvas[
                top*output_scale:(top+core_size)*output_scale,
                left*output_scale:(left+core_size)*output_scale,
            ] = output

    result = result_canvas[
        :model_height*output_scale,
        :model_width*output_scale,
    ]

    # Real-ESRGAN会放大四倍，这里恢复为用户原图宽高，避免生成超大图片
    result = cv2.resize(
        result,
        (original_width, original_height),
        interpolation=cv2.INTER_LANCZOS4,
    )

    # Real-ESRGAN输出是RGB，直接交给Pillow保存，避免红蓝通道互换
    return image_response(result, dpi)


# 当直接运行这个文件时，在所有网卡的8081端口启动图片接口
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)
