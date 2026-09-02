# API文档

## 必读：

1. 调用接口需要使用 `POST` 请求，然后通过 `multipart/form-data` 的方式在同一次请求中上传 `input_image` 图片和其它参数
2. 所有接口都必须根据 HTTP 状态码判断成功和失败：`200` 代表成功，`404` 代表失败
3. 失败时：所有接口统一返回 JSON；
4. 成功时：`/checkImg` 接口返回 JSON，其它接口返回图片二进制



因为鉴黄接口比较特殊，所以鉴黄接口与其它接口的返回格式不一样

### 鉴黄接口状态码

| HTTP 状态码 | 含义 | 响应类型 |
| --- | --- | --- |
| `200` | 请求成功；返回鉴黄分类分数 | `application/json` |
| `404` | 请求失败；返回错误信息，如：参数错误、模型缺失等 | `application/json` |

### 其它接口状态码

| HTTP 状态码 | 含义 | 响应类型 |
| --- | --- | --- |
| `200` | 请求成功；返回图片二进制 | `image/png`、`image/jpeg` 或 `image/gif` |
| `404`       | 请求失败；返回错误信息，如：参数错误、模型缺失等 | `application/json` |

---





## 1. 图片鉴黄

返回五项图片分类分数

### 请求 URL

`POST http://127.0.0.1:8081/checkImg`

### 入参

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input_image` | `UploadFile` | 是 | - | 待检测图片，接口按 RGB 三通道图像读取 |



### 返回

| HTTP 状态 | Content-Type | 内容 |
| --- | --- | --- |
| `200` | `application/json` | 返回五项图片分类分数 |

返回示例：

```json
{
  "code": 200,
  "msg": "检测成功",
  "data": {
    "drawings": 0.001,
    "hentai": 0.002,
    "neutral": 0.995,
    "porn": 0.001,
    "sexy": 0.001
  }
}
```

`drawings`、`hentai`、`neutral`、`porn`、`sexy` 分别表示绘画、色情动漫、中性、色情和性感分数；接口只返回分数，不执行拦截





## 2. 生成证件照

从原图生成透明背景的标准证件照或高清证件照

### 请求 URL

`POST http://127.0.0.1:8081/idphoto`

### 入参

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input_image` | `UploadFile` | 是 | - | 单人人像图片，接口按 RGB 三通道图像读取 |
| `height` | `int` | 否 | `413` | 标准证件照高度，单位为像素 |
| `width` | `int` | 否 | `295` | 标准证件照宽度，单位为像素 |
| `human_matting_model` | `str` | 否 | `ppMattingV2` | 人像分割模型，可选：`ppMattingV2`、`hivisionModnet`、`modnetPhotographic`、`rmbg`、`silueta`、`birefnet` |
| `face_detect_model` | `str` | 否 | `yunet` | 人脸检测模型，可选：`yunet`、`mtcnn`、`retinaface` |
| `hd` | `bool` | 否 | `true` | `true` 返回高清照，`false` 返回标准尺寸照片 |
| `dpi` | `int` | 否 | `300` | 输出图片 DPI |
| `face_align` | `bool` | 否 | `false` | 是否自动矫正头部倾斜 |
| `head_measure_ratio` | `float` | 否 | `0.2` | 人脸面积占照片面积的比例 |
| `head_height_ratio` | `float` | 否 | `0.45` | 人脸中心在照片高度中的位置比例 |
| `top_distance_max` | `float` | 否 | `0.12` | 头顶到照片顶部的最大距离比例 |
| `top_distance_min` | `float` | 否 | `0.10` | 头顶到照片顶部的最小距离比例 |
| `whitening_strength` | `int` | 否 | `0` | 美白强度，`0` 表示不处理 |
| `brightness_strength` | `float` | 否 | `0` | 亮度，`0` 表示不处理 |
| `contrast_strength` | `float` | 否 | `0` | 对比度，`0` 表示不处理 |
| `sharpen_strength` | `float` | 否 | `0` | 锐化强度，`0` 表示不处理 |
| `saturation_strength` | `float` | 否 | `0` | 饱和度，`0` 表示不处理 |



### 返回

| HTTP 状态 | Content-Type | 内容 |
| --- | --- | --- |
| `200` | `image/png` | 返回透明背景图片 |

---





## 3. 证件照添加背景

给透明证件照添加纯色或渐变背景

### 请求 URL

`POST http://127.0.0.1:8081/add_background`

### 入参

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input_image` | `UploadFile` | 是 | - | 透明背景图片 |
| `color` | `str` | 否 | `#000000` | 背景颜色，必须使用完整的 `#RRGGBB` 格式 |
| `dpi` | `int` | 否 | `300` | 输出图片 DPI |
| `render` | `int` | 否 | `0` | 背景样式：`0` 纯色、`1` 上下渐变、`2` 中心渐变 |
| `kb` | `int` | 否 | `None` | 目标文件大小，单位为 KB；填写后返回 JPEG |



### 返回

| 条件 | HTTP 状态 | Content-Type | 内容 |
| --- | --- | --- | --- |
| 不传入kb | `200` | `image/png` | 返回 PNG 图片 |
| 传入kb | `200` | `image/jpeg` | 返回 JPEG 图片 |

---





## 4. 证件照换装

给透明证件照更换内置服装，并按照当前背景参数生成完整成片

### 请求 URL

`POST http://127.0.0.1:8081/change_clothes`

### 入参

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input_image` | `UploadFile` | 是 | - | 透明背景图片 |
| `clothes_category` | `int` | 否 | `0` | 服装分类：`0` 不换装、`1` 男装、`2` 女装、`3` 儿童装 |
| `clothes_id` | `int` | 否 | `0` | 服装编号；男装`1～17`，女装`1～17` ，儿童装 `1～9`。素材在/tools/clothes/assets/里面 |
| `clothes_face_detect_model` | `str` | 是 | - | 换装人脸检测模型，只支持 `yunet` |
| `clothes_parsing_model` | `str` | 是 | - | 换装人体解析模型，只支持 `selfieMulticlass` |
| `color` | `str` | 否 | `#ffffff` | 背景颜色，必须使用完整的 `#RRGGBB` 格式 |
| `dpi` | `int` | 否 | `300` | 输出图片 DPI |
| `render` | `int` | 否 | `0` | 背景样式：`0` 纯色、`1` 上下渐变、`2` 中心渐变 |
| `kb` | `int` | 否 | `None` | 目标文件大小，单位为 KB；填写后返回 JPEG |



### 返回

| 条件 | HTTP 状态 | Content-Type | 内容 |
| --- | --- | --- | --- |
| 不传入kb | `200` | `image/png` | 返回 PNG 图片 |
| 传入kb | `200` | `image/jpeg` | 返回 JPEG 图片 |

---





## 5. 生成美式证件照

从人物原图完成抠图、裁切并合成美式背景

### 请求 URL

`POST http://127.0.0.1:8081/american_idphoto`

### 入参

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input_image` | `UploadFile` | 是 | - | 单人人像图片，接口按 RGB 三通道图像读取 |
| `height` | `int` | 否 | `600` | 输出图片高度，单位为像素 |
| `width` | `int` | 否 | `600` | 输出图片宽度，单位为像素 |
| `human_matting_model` | `str` | 否 | `ppMattingV2` | 人像分割模型，可选：`ppMattingV2`、`hivisionModnet`、`modnetPhotographic`、`rmbg`、`silueta`、`birefnet` |
| `face_detect_model` | `str` | 否 | `yunet` | 人脸检测模型，可选：`yunet`、`mtcnn`、`retinaface` |
| `face_align` | `bool` | 否 | `true` | 是否自动矫正头部倾斜 |
| `background_image` | `UploadFile` | 否 | `None` | 自定义背景图片，上传后优先使用 |
| `background` | `str` | 否 | `american` | 内置背景，目前支持传入 `american` |
| `dpi` | `int` | 否 | `300` | 输出图片 DPI |
| `kb` | `int` | 否 | `None` | 目标文件大小，单位为 KB |



### 返回

| 条件 | HTTP 状态 | Content-Type | 内容 |
| --- | --- | --- | --- |
| 不传入kb | `200` | `image/png` | 返回 PNG 图片 |
| 传入kb | `200` | `image/jpeg` | 返回 JPEG 图片 |

---





## 6. 证件照排版

将一张已经完成背景合成的证件照排入相纸画布

### 请求 URL

`POST http://127.0.0.1:8081/generate_layout_photos`

### 入参

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input_image` | `UploadFile` | 是 | - | 已添加背景的证件照，尺寸应与本次传入的 `width × height` 一致 |
| `height` | `int` | 否 | `413` | 单张证件照高度，单位为像素 |
| `width` | `int` | 否 | `295` | 单张证件照宽度，单位为像素 |
| `dpi` | `int` | 否 | `300` | 输出图片 DPI |
| `layout_size` | `str` | 否 | `None` | 预设画布名称，可选：`six_inch`、`five_inch`、`a4`、`three_r`、`four_r` |
| `layout_height` | `int` | 否 | `None` | 自定义画布高度，单位为像素，必须与 `layout_width` 同时传入 |
| `layout_width` | `int` | 否 | `None` | 自定义画布宽度，单位为像素，必须与 `layout_height` 同时传入 |
| `crop_line` | `bool` | 否 | `false` | 是否添加裁切线 |
| `kb` | `int` | 否 | `None` | 目标文件大小，单位为 KB |

#### 注意：

传入 `layout_size` 时，不能传入 `layout_height` 和 `layout_width`

传入 `layout_height` 和 `layout_width` 时，不能传入 `layout_size`

三项都不传入时，默认使用 `six_inch`



预设画布尺寸是：

| `layout_size` | 中文名称 | 画布尺寸（高度 px × 宽度 px） |
| --- | --- | --- |
| `six_inch` | 六寸 | `1205 × 1795` |
| `five_inch` | 五寸 | `1051 × 1500` |
| `a4` | A4 | `2479 × 3508` |
| `three_r` | 3R | `1051 × 1500` |
| `four_r` | 4R | `1205 × 1795` |



### 返回

| 条件 | HTTP 状态 | Content-Type | 内容 |
| --- | --- | --- | --- |
| 不传入kb | `200` | `image/png` | 返回 PNG 图片 |
| 传入kb | `200` | `image/jpeg` | 返回 JPEG 图片 |

---





## 7. 生成模板照

从人物原图生成证件照，再填充背景色并合成社交媒体模板

### 请求 URL

`POST http://127.0.0.1:8081/generate_template_photos`

### 入参

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input_image` | `UploadFile` | 是 | - | 单人人像图片，接口按 RGB 三通道图像读取 |
| `height` | `int` | 否 | `413` | 模板内证件照高度，单位为像素 |
| `width` | `int` | 否 | `295` | 模板内证件照宽度，单位为像素 |
| `human_matting_model` | `str` | 否 | `ppMattingV2` | 人像分割模型，可选：`ppMattingV2`、`hivisionModnet`、`modnetPhotographic`、`rmbg`、`silueta`、`birefnet` |
| `face_detect_model` | `str` | 否 | `yunet` | 人脸检测模型，可选：`yunet`、`mtcnn`、`retinaface` |
| `face_align` | `bool` | 否 | `true` | 是否自动矫正头部倾斜 |
| `template_name` | `str` | 否 | `template_1` | 模板名称，可选：`template_1`、`template_2` |
| `color` | `str` | 否 | `#438edb` | 模板内证件照的背景颜色，必须使用完整的 `#RRGGBB` 格式 |
| `dpi` | `int` | 否 | `300` | 输出图片 DPI |
| `kb` | `int` | 否 | `None` | 目标文件大小，单位为 KB |



### 返回

| 条件 | HTTP 状态 | Content-Type | 内容 |
| --- | --- | --- | --- |
| 不传入kb | `200` | `image/png` | 返回 PNG 图片 |
| 传入kb | `200` | `image/jpeg` | 返回 JPEG 图片 |

---





## 8. 图片抠图

从图片中抠出人像或主体，返回透明背景图片

### 请求 URL

`POST http://127.0.0.1:8081/human_matting`

### 入参

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input_image` | `UploadFile` | 是 | - | 待抠图图片，接口按 RGB 三通道图像读取 |
| `human_matting_model` | `str` | 否 | `ppMattingV2` | 分割模型，可选：`ppMattingV2`、`hivisionModnet`、`modnetPhotographic`、`rmbg`、`silueta`、`birefnet` |
| `dpi` | `int` | 否 | `300` | 输出图片 DPI |



### 返回

| HTTP 状态 | Content-Type | 内容 |
| --- | --- | --- |
| `200` | `image/png` | 返回透明背景抠图；大图会等比例缩小到最长边 `2000px` |

---





## 9. 图片添加水印

给图片添加斜纹平铺水印或居中水印

### 请求 URL

`POST http://127.0.0.1:8081/watermark`

### 入参

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input_image` | `UploadFile` | 是 | - | 待添加水印的图片，支持灰度、RGB 和 RGBA 图像 |
| `text` | `str` | 是 | - | 水印文字 |
| `style` | `str` | 否 | `striped` | 水印样式：`striped` 平铺、`central` 居中 |
| `angle` | `int` | 否 | `30` | 水印旋转角度 |
| `color` | `str` | 否 | `#171717` | 水印颜色，必须使用完整的 `#RRGGBB` 格式 |
| `opacity` | `float` | 否 | `0.30` | 水印不透明度，建议使用 `0～1` |
| `size` | `int` | 否 | `40` | 水印基准字号，会根据图片大小缩放 |
| `space` | `int` | 否 | `120` | 平铺水印基准间距，会根据图片大小缩放 |
| `dpi` | `int` | 否 | `300` | 输出图片 DPI |
| `kb` | `int` | 否 | `None` | 目标文件大小，单位为 KB；填写后返回 JPEG |



### 返回

| 条件 | HTTP 状态 | Content-Type | 说明 |
| --- | --- | --- | --- |
| 不传入kb | `200` | `image/png` | 返回 PNG 图片，保留透明通道 |
| 传入kb | `200` | `image/jpeg` | 返回 JPEG 图片，不保留透明通道 |

---





## 10. 图片 KB 压缩

将图片重新编码成 JPEG，并尽量压缩到指定 KB

### 请求 URL

`POST http://127.0.0.1:8081/set_kb`

### 入参

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input_image` | `UploadFile` | 是 | - | 待压缩图片，接口按 RGB 三通道图像读取 |
| `dpi` | `int` | 否 | `300` | 输出图片 DPI |
| `kb` | `int` | 否 | `50` | 目标文件大小，单位为 KB，必须大于 `0` |



### 返回

| HTTP 状态 | Content-Type | 内容 |
| --- | --- | --- |
| `200` | `image/jpeg` | 返回 JPEG 图片 |

---





## 11. 图片转换格式

将图片转换为 JPG、JPEG、PNG 或静态 GIF

### 请求 URL

`POST http://127.0.0.1:8081/convert_image_format`

### 入参

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input_image` | `UploadFile` | 是 | - | 待转换图片 |
| `target_format` | `str` | 是 | - | 目标格式，可选：`jpg`、`jpeg`、`png`、`gif` |



### 返回

| 条件 | HTTP 状态 | Content-Type | 内容 |
| --- | --- | --- | --- |
| `target_format` 为 `jpg` 或 `jpeg`时 | `200` | `image/jpeg` | 返回 JPEG 图片，不保留透明通道 |
| `target_format` 为 `png`时 | `200` | `image/png` | 返回 PNG 图片，保留透明通道 |
| `target_format` 为 `gif`时 | `200` | `image/gif` | 返回静态单帧 GIF 图片 |

---





## 12. 图片转动漫风

使用 DCT-Net 将图片转换成立体动漫风格

### 请求 URL

`POST http://127.0.0.1:8081/cartoon`

### 入参

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input_image` | `UploadFile` | 是 | - | 待处理图片，接口按 RGB 三通道图像读取 |
| `cartoon_model` | `str` | 否 | `cartoon` | 动漫风模型，目前支持 `cartoon` |



### 返回

| HTTP 状态 | Content-Type | 内容 |
| --- | --- | --- |
| `200` | `image/png` | 返回 PNG 图片 |

---





## 13. 黑白照片上色

使用 DDColor ModelScope 为黑白照片生成颜色

### 请求 URL

`POST http://127.0.0.1:8081/colourizeImg`

### 入参

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input_image` | `UploadFile` | 是 | - | 待上色图片，接口按 RGB 三通道图像读取 |
| `colourize_model` | `str` | 否 | `ddcolor` | 黑白照片上色模型，目前支持 `ddcolor` |
| `dpi` | `int` | 否 | `300` | 输出图片 DPI |



### 返回

| HTTP 状态 | Content-Type | 内容 |
| --- | --- | --- |
| `200` | `image/png` | 返回 PNG 图片 |

---





## 14. 模糊图片变清晰

使用 Real-ESRGAN 增强模糊或低清图片，最终保持原图宽高

### 请求 URL

`POST http://127.0.0.1:8081/deblur`

### 入参

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input_image` | `UploadFile` | 是 | - | 模糊或低清图片，接口按 RGB 三通道图像读取 |
| `deblur_model` | `str` | 否 | `realEsrgan` | 图片清晰度增强模型，目前支持 `realEsrgan` |
| `dpi` | `int` | 否 | `300` | 输出图片 DPI |



### 返回

| HTTP 状态 | Content-Type | 内容 |
| --- | --- | --- |
| `200` | `image/png` | 返回 PNG 图片 |

---




## 15. 情侣红底照

检测照片中的两个人，保留原图尺寸并合成传入的红色背景

### 请求 URL

`POST http://127.0.0.1:8081/couple_red_photo`

### 入参

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input_image` | `UploadFile` | 是 | - | 必须包含两个人的原始照片 |
| `color` | `str` | 否 | `#5C1117` | `#5C1117`生成酒红渐变背景，也可选择`#D9001B`、`#D12C25`、`#FF2121`、`#FF0000`、`#FF4C00`、`#9D2933`、`#ED333D`纯色背景 |
| `human_matting_model` | `str` | 否 | `ppMattingV2` | 人像分割模型，可选：`ppMattingV2`、`hivisionModnet`、`modnetPhotographic`、`rmbg`、`silueta`、`birefnet` |
| `face_detect_model` | `str` | 否 | `yunet` | 人脸检测模型，可选：`yunet`、`mtcnn`、`retinaface` |
| `dpi` | `int` | 否 | `300` | 输出图片 DPI |

### 返回

| HTTP 状态 | Content-Type | 内容 |
| --- | --- | --- |
| `200` | `image/png` | 返回保持原图宽高的情侣红底照 |
| `404` | `application/json` | 返回人脸数量、颜色、模型或图片处理错误 |
