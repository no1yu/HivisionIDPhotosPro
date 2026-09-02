import cv2
import numpy as np


def _find_pupil(landmarks, image):
    """在眼部关键点范围内找到瞳孔中心，找不到时交给调用方使用关键点均值。"""
    height, width = image.shape[:2]
    max_x = int(landmarks[:, 0].max())
    min_x = int(landmarks[:, 0].min())
    max_y = int(landmarks[:, 1].max())
    min_y = int(landmarks[:, 1].min())

    # 当眼部区域已经超出图片或没有有效宽高时，放弃查找瞳孔
    if min_y >= max_y or min_x >= max_x or min_y < 0 or min_x < 0 or max_y > height or max_x > width:
        return None

    eye_image = cv2.cvtColor(image[min_y:max_y, min_x:max_x], cv2.COLOR_BGR2GRAY)
    eye_image = cv2.equalizeHist(eye_image)
    eye_mask = cv2.fillConvexPoly(
        np.zeros_like(eye_image),
        (landmarks-np.array([min_x, min_y])).astype(np.int32),
        1,
    )
    _, threshold = cv2.threshold(eye_image, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    pupil_y, pupil_x = np.where((1-threshold/255.0)*eye_mask > 0.5)

    # 当眼部区域没有找到暗色像素时，使用眼部区域中心
    if pupil_x.size == 0:
        return ((max_x+min_x)/2, (max_y+min_y)/2)

    pupil_x.sort()
    pupil_y.sort()
    return (pupil_x[pupil_x.size//2]+min_x, pupil_y[pupil_y.size//2]+min_y)


def _nonreflective_similarity(source_points, target_points):
    """按照DCT-Net官方对齐代码计算不带镜像的相似变换。"""
    target_x = target_points[:, 0].reshape((-1, 1))
    target_y = target_points[:, 1].reshape((-1, 1))
    matrix = np.vstack((
        np.hstack((target_x, target_y, np.ones((target_points.shape[0], 1)), np.zeros((target_points.shape[0], 1)))),
        np.hstack((target_y, -target_x, np.zeros((target_points.shape[0], 1)), np.ones((target_points.shape[0], 1)))),
    ))
    source = np.vstack((
        source_points[:, 0].reshape((-1, 1)),
        source_points[:, 1].reshape((-1, 1)),
    ))
    result = np.linalg.lstsq(matrix, source, rcond=-1)[0].reshape(-1)
    transform_inverse = np.array([
        [result[0], -result[1], 0],
        [result[1], result[0], 0],
        [result[2], result[3], 1],
    ])
    transform = np.linalg.inv(transform_inverse)
    transform[:, 2] = np.array([0, 0, 1])
    return transform, transform_inverse


def _similarity_transform(source_points, target_points):
    """从正向和镜像相似变换中选择误差更小的一组矩阵。"""
    transform, transform_inverse = _nonreflective_similarity(source_points, target_points)
    reflected_points = target_points.copy()
    reflected_points[:, 0] *= -1
    reflected_transform, _ = _nonreflective_similarity(source_points, reflected_points)
    reflected_transform = reflected_transform@np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    source_augmented = np.hstack((source_points, np.ones((source_points.shape[0], 1))))
    normal_error = np.linalg.norm((source_augmented@transform)[:, :2]-target_points)
    reflected_error = np.linalg.norm((source_augmented@reflected_transform)[:, :2]-target_points)

    # 当镜像变换误差更小时，使用镜像变换及其逆矩阵
    if reflected_error < normal_error:
        return reflected_transform[:, :2].T, np.linalg.inv(reflected_transform)[:, :2].T

    return transform[:, :2].T, transform_inverse[:, :2].T


def _detect_landmarks(image, detector_session, keypoints_session):
    """使用DCT-Net配套检测器找到图片中的人脸和68个关键点。"""
    height, width = image.shape[:2]
    scale = 512.0/max(height, width)
    resized_image = cv2.resize(image, None, fx=scale, fy=scale)
    detector_image = np.zeros((512, 512, 3), dtype=np.float32)+np.array([123, 116, 103], dtype=np.float32)
    detector_image[:resized_image.shape[0], :resized_image.shape[1]] = resized_image
    detector_inputs = detector_session.get_inputs()
    boxes, scores, box_count = detector_session.run(
        None,
        {
            detector_inputs[0].name: detector_image[None],
            detector_inputs[1].name: np.array(False),
        },
    )
    box_count = int(np.asarray(box_count).reshape(-1)[0])
    boxes = boxes[0][:box_count]
    scores = scores[0][:box_count]
    boxes = boxes[scores > 0.8]*max(height, width)

    # 当图片没有检测到可信人脸时，只返回整图动漫效果
    if boxes.shape[0] == 0:
        return []

    boxes = boxes[:, [1, 0, 3, 2]]
    box_areas = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    boxes = boxes[np.argsort(box_areas)[::-1][:10]]
    keypoints_inputs = keypoints_session.get_inputs()
    landmarks = []

    for box in boxes:
        box_width = box[2]-box[0]
        box_height = box[3]-box[1]

        # 当人脸宽高都不超过60像素时，跳过无法稳定定位五官的小脸
        if box_width <= 60 and box_height <= 60:
            continue

        border = int(max(box_width, box_height))
        bordered_image = cv2.copyMakeBorder(
            image,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=(123, 116, 103),
        )
        moved_box = box.copy()+border
        edge = 1.4*box_width
        center_x = (moved_box[0]+moved_box[2])//2
        center_y = (moved_box[1]+moved_box[3])//2
        crop_box = np.array([
            center_x-edge//2,
            center_y-edge//2,
            center_x+edge//2,
            center_y+edge//2,
        ]).astype(np.int32)
        cropped_image = bordered_image[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
        crop_height, crop_width = cropped_image.shape[:2]
        keypoint_image = cv2.resize(cropped_image, (160, 160)).astype(np.float32)
        prediction = keypoints_session.run(
            None,
            {
                keypoints_inputs[0].name: keypoint_image[None],
                keypoints_inputs[1].name: np.array(False),
            },
        )[0][0][:136].reshape((-1, 2))
        prediction[:, 0] = prediction[:, 0]*crop_width+crop_box[0]-border
        prediction[:, 1] = prediction[:, 1]*crop_height+crop_box[1]-border
        landmarks.append(prediction.astype(np.int32).astype(np.float32))

    landmarks.sort(
        key=lambda points: (points[:, 0].max()-points[:, 0].min())*(points[:, 1].max()-points[:, 1].min()),
        reverse=True,
    )
    return landmarks


def cartoonize(image, cartoon_session, detector_session, keypoints_session, alpha_path):
    """按DCT-Net官方流程处理整图，并对检测到的人脸进行单独增强和融合。"""
    original_height, original_width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # 当图片短边超过720像素时，等比例缩小模型工作图以控制CPU处理量
    if min(height, width) > 720:
        # 当图片是竖图时，把宽度缩到720像素
        if height > width:
            height, width = int(720*height/width), 720
        else:
            height, width = 720, int(720*width/height)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    image_bgr = image[:, :, ::-1]

    # 当宽高不全是16的倍数时，按照DCT-Net官方规则让两个方向都补到下一个倍数
    if height%16 != 0 or width%16 != 0:
        padded_height = (height//16+1)*16
        padded_width = (width//16+1)*16
    else:
        padded_height = height
        padded_width = width
    padded_image = np.full((padded_height, padded_width, 3), 255, dtype=np.float32)
    padded_image[:height, :width] = image_bgr
    cartoon_input_name = cartoon_session.get_inputs()[0].name
    result = cartoon_session.run(None, {cartoon_input_name: padded_image})[0][:height, :width].astype(np.float32)
    landmarks = _detect_landmarks(image, detector_session, keypoints_session)
    alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
    alpha = cv2.resize(alpha, (288, 288), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0

    for face_landmarks in landmarks:
        left_eye = _find_pupil(face_landmarks[36:42], image_bgr)
        right_eye = _find_pupil(face_landmarks[42:48], image_bgr)

        # 当任意一只眼睛无法定位瞳孔时，两只眼睛都使用关键点均值完成对齐
        if left_eye is None or right_eye is None:
            left_eye = face_landmarks[36:42].mean(axis=0)
            right_eye = face_landmarks[42:48].mean(axis=0)

        facial_points = np.float32([
            left_eye,
            right_eye,
            face_landmarks[30],
            face_landmarks[48],
            face_landmarks[54],
        ])
        reference_points = np.array([
            [39.29459953, 52.69630051],
            [74.53179932, 52.50139999],
            [57.02519989, 72.73660278],
            [42.54930115, 93.3655014],
            [71.72990036, 93.20410156],
        ]).astype(np.float32)
        reference_points = (reference_points-56)*0.75+56
        reference_points *= 288/112
        transform, transform_inverse = _similarity_transform(facial_points, reference_points)
        head_image = cv2.warpAffine(image, transform, (288, 288), borderValue=(255, 255, 255))
        head_result = cartoon_session.run(
            None,
            {cartoon_input_name: head_image[:, :, ::-1].astype(np.float32)},
        )[0]
        head_result = cv2.warpAffine(head_result, transform_inverse, (width, height), borderValue=(0, 0, 0))
        face_alpha = cv2.warpAffine(alpha, transform_inverse, (width, height), borderValue=0)[:, :, None]
        result = face_alpha*head_result+(1-face_alpha)*result

    result = cv2.resize(result, (original_width, original_height), interpolation=cv2.INTER_AREA)
    return np.clip(result, 0, 255).round().astype(np.uint8)
