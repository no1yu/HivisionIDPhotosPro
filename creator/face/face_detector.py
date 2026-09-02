#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2024/9/5 19:32
@File: face_detector.py
@IDE: pycharm
@Description:
    人脸检测器
"""
try:
    from mtcnnruntime import MTCNN
except ImportError:
    raise ImportError(
        "Please install mtcnn-runtime by running `pip install mtcnn-runtime`"
    )
from ..context import Context
from tools.exceptions import FaceError, ModelNotFoundError
from tools.model_files import get_model_files
from .retinaface.inference import retinaface_detect_faces
import cv2
import numpy as np
from threading import Lock


mtcnn = None
RETINAFACE_SESS = None
YUNET_DETECTOR = None
YUNET_LOCK = Lock()


def _face_result(rectangle, landmarks, score):
    """把不同人脸检测模型的输出统一为矩形、五点坐标和置信度"""
    return {
        "rectangle": tuple(float(value) for value in rectangle),
        "landmarks": np.asarray(landmarks, dtype=np.float32).reshape(-1, 2),
        "score": float(score),
    }


def detect_faces_mtcnn(image: np.ndarray):
    """使用MTCNN检测图片中的全部人脸"""
    global mtcnn

    if mtcnn is None:
        mtcnn = MTCNN()
    faces, landmarks = mtcnn.detect(image, thresholds=[0.8, 0.8, 0.8])
    results = []
    for index, face in enumerate(faces):
        left, top, right, bottom = face[:4]
        face_landmarks = landmarks[index]
        points = np.column_stack((face_landmarks[:5], face_landmarks[5:]))
        results.append(_face_result(
            (left, top, right - left + 1, bottom - top + 1),
            points,
            face[4] if len(face) > 4 else 1,
        ))
    return results


def detect_faces_retinaface(image: np.ndarray):
    """使用RetinaFace检测图片中的全部人脸"""
    global RETINAFACE_SESS

    model_path, = get_model_files("retinaface", "retinaface-resnet50.onnx")
    faces, session = retinaface_detect_faces(
        image,
        model_path,
        sess=RETINAFACE_SESS,
    )
    RETINAFACE_SESS = session
    results = []
    for face in faces:
        left, top, right, bottom = face[:4]
        results.append(_face_result(
            (left, top, right - left + 1, bottom - top + 1),
            np.asarray(face[5:15]).reshape(5, 2),
            face[4],
        ))
    return results


def detect_face_mtcnn(ctx: Context, scale: int = 2):
    """
    基于MTCNN模型的人脸检测处理器，只进行人脸数量的检测
    :param ctx: 上下文，此时已获取到原始图和抠图结果，但是我们只需要原始图
    :param scale: 最大边长缩放比例，原图:缩放图 = 1:scale
    :raise FaceError: 人脸检测错误，多个人脸或者没有人脸
    """
    global mtcnn
    if mtcnn is None:
        mtcnn = MTCNN()
    image = cv2.resize(
        ctx.origin_image,
        (ctx.origin_image.shape[1] // scale, ctx.origin_image.shape[0] // scale),
        interpolation=cv2.INTER_AREA,
    )
    # landmarks 是 5 个关键点，分别是左眼、右眼、鼻子、左嘴角、右嘴角，
    faces, landmarks = mtcnn.detect(image, thresholds=[0.8, 0.8, 0.8])

    # print(len(faces))
    if len(faces) != 1:
        # 保险措施，如果检测到多个人脸或者没有人脸，用原图再检测一次
        faces, landmarks = mtcnn.detect(ctx.origin_image)
    else:
        # 如果只有一个人脸，将人脸坐标放大
        for item, param in enumerate(faces[0]):
            faces[0][item] = param * 2
    if len(faces) != 1:
        raise FaceError("Expected 1 face, but got {}".format(len(faces)), len(faces))

    # 计算人脸坐标
    left = faces[0][0]
    top = faces[0][1]
    width = faces[0][2] - left + 1
    height = faces[0][3] - top + 1
    ctx.face["rectangle"] = (left, top, width, height)

    # 根据landmarks计算人脸偏转角度，以眼睛为标准，计算的人脸偏转角度，用于人脸矫正
    # 示例landmarks [106.37181  150.77415  127.21012  108.369156 144.61522  105.24723 107.45625  133.62355  151.24269  153.34407 ]
    landmarks = landmarks[0]
    left_eye = np.array([landmarks[0], landmarks[5]])
    right_eye = np.array([landmarks[1], landmarks[6]])
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    roll_angle = np.degrees(np.arctan2(dy, dx))

    ctx.face["roll_angle"] = roll_angle


def detect_face_retinaface(ctx: Context):
    """
    基于RetinaFace模型的人脸检测处理器，只进行人脸数量的检测
    :param ctx: 上下文，此时已获取到原始图和抠图结果，但是我们只需要原始图
    :raise FaceError: 人脸检测错误，多个人脸或者没有人脸
    """
    from time import time

    global RETINAFACE_SESS
    model_path, = get_model_files("retinaface", "retinaface-resnet50.onnx")

    if RETINAFACE_SESS is None:
        # 计算用时
        tic = time()
        faces_dets, sess = retinaface_detect_faces(
            ctx.origin_image,
            model_path,
            sess=None,
        )
        RETINAFACE_SESS = sess
    else:
        tic = time()
        faces_dets, _ = retinaface_detect_faces(
            ctx.origin_image,
            model_path,
            sess=RETINAFACE_SESS,
        )

    faces_num = len(faces_dets)
    faces_landmarks = []
    for face_det in faces_dets:
        faces_landmarks.append(face_det[5:])

    if faces_num != 1:
        raise FaceError("Expected 1 face, but got {}".format(faces_num), faces_num)
    face_det = faces_dets[0]
    ctx.face["rectangle"] = (
        face_det[0],
        face_det[1],
        face_det[2] - face_det[0] + 1,
        face_det[3] - face_det[1] + 1,
    )

    # 计算roll_angle
    face_landmarks = faces_landmarks[0]
    # print("face_landmarks", face_landmarks)
    left_eye = np.array([face_landmarks[0], face_landmarks[1]])
    right_eye = np.array([face_landmarks[2], face_landmarks[3]])
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    roll_angle = np.degrees(np.arctan2(dy, dx))
    ctx.face["roll_angle"] = roll_angle


def detect_faces_yunet(image: np.ndarray):
    """使用YuNet检测图片中的人脸并返回独立的检测结果。"""
    global YUNET_DETECTOR

    model_path, = get_model_files("yunet", "face_detection_yunet_2026may.onnx")
    image_height, image_width = image.shape[:2]

    # FaceDetectorYN会保存输入尺寸，首次加载及每次修改尺寸都需要串行处理
    with YUNET_LOCK:
        if YUNET_DETECTOR is None:
            YUNET_DETECTOR = cv2.FaceDetectorYN.create(
                model_path,
                "",
                (image_width, image_height),
                0.8,
                0.3,
                5000,
            )
        else:
            YUNET_DETECTOR.setInputSize((image_width, image_height))
        _, faces = YUNET_DETECTOR.detect(image)
        faces = None if faces is None else faces.copy()
    return faces


def detect_faces(image: np.ndarray, model: str):
    """按照模型配置检测全部人脸并返回统一结果"""
    if model == "yunet":
        faces = detect_faces_yunet(image)
        if faces is None:
            return []
        return [
            _face_result(
                (face[0], face[1], face[2], face[3]),
                np.asarray(face[4:14]).reshape(5, 2),
                face[14],
            )
            for face in faces
        ]
    if model == "mtcnn":
        return detect_faces_mtcnn(image)
    if model == "retinaface":
        return detect_faces_retinaface(image)
    raise ModelNotFoundError()


def detect_face_yunet(ctx: Context):
    """使用YuNet检测唯一人脸，并根据双眼关键点计算头部倾斜角度。"""
    faces = detect_faces_yunet(ctx.origin_image)

    faces_num = 0 if faces is None else len(faces)
    if faces_num != 1:
        raise FaceError("Expected 1 face, but got {}".format(faces_num), faces_num)

    face = faces[0]
    ctx.face["rectangle"] = (
        float(face[0]),
        float(face[1]),
        float(face[2]),
        float(face[3]),
    )

    # YuNet关键点顺序为右眼、左眼、鼻尖、右嘴角、左嘴角
    right_eye = np.array([face[4], face[5]])
    left_eye = np.array([face[6], face[7]])
    dy = left_eye[1] - right_eye[1]
    dx = left_eye[0] - right_eye[0]
    ctx.face["roll_angle"] = float(np.degrees(np.arctan2(dy, dx)))

