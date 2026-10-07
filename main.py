import cv2
import numpy as np
import os

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/face_detection_yunet_2026may.onnx"

FACE_SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3

BACKGROUND_PASS_SCORE = 55.0

WHITE_BRIGHTNESS = 190
WHITE_COLOR_DIFFERENCE = 35

MAX_HORIZONTAL_OFFSET = 0.12

MIN_FACE_WIDTH_RATIO = 0.15
MAX_FACE_WIDTH_RATIO = 0.50

MIN_TOP_MARGIN_RATIO = 0.08
MAX_TOP_MARGIN_RATIO = 0.40


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Student Photo Analyzer"
)


# ============================================================
# LOAD YUNET
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YuNet model not found: {MODEL_PATH}"
    )


detector = cv2.FaceDetectorYN_create(
    MODEL_PATH,
    "",
    (320, 320),
    FACE_SCORE_THRESHOLD,
    NMS_THRESHOLD,
    5000
)


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(image):

    height, width = image.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, detections = detector.detect(image)

    if detections is None:
        return []

    return detections


# ============================================================
# WHITE PIXEL CHECK
# ============================================================

def calculate_white_percentage(
    region,
    mask=None
):

    pixels = region.reshape(-1, 3)

    if mask is not None:

        mask_flat = mask.reshape(-1)

        pixels = pixels[
            mask_flat > 0
        ]

    if len(pixels) == 0:
        return None

    brightness = np.mean(
        pixels,
        axis=1
    )

    brightest = np.max(
        pixels,
        axis=1
    )

    darkest = np.min(
        pixels,
        axis=1
    )

    difference = (
        brightest - darkest
    )

    white = (
        (brightness >= WHITE_BRIGHTNESS)
        &
        (difference <= WHITE_COLOR_DIFFERENCE)
    )

    return (
        np.sum(white)
        /
        len(white)
    ) * 100


# ============================================================
# FACE MASK
# ============================================================

def create_face_mask(
    image,
    detections
):

    height, width = image.shape[:2]

    mask = np.ones(
        (height, width),
        dtype=np.uint8
    ) * 255

    for face in detections:

        x = int(face[0])
        y = int(face[1])
        w = int(face[2])
        h = int(face[3])

        padding_x = int(w * 0.7)
        padding_y = int(h * 1.0)

        x1 = max(
            0,
            x - padding_x
        )

        y1 = max(
            0,
            y - padding_y
        )

        x2 = min(
            width,
            x + w + padding_x
        )

        y2 = min(
            height,
            y + h + padding_y
        )

        mask[
            y1:y2,
            x1:x2
        ] = 0

    return mask


# ============================================================
# BACKGROUND ANALYSIS
# ============================================================

def analyze_background(
    image,
    detections
):

    height, width = image.shape[:2]

    mask = create_face_mask(
        image,
        detections
    )

    border = max(
        10,
        int(
            min(height, width) * 0.08
        )
    )

    regions = [
        (
            image[:border, :],
            mask[:border, :]
        ),
        (
            image[
                height - border:,
                :
            ],
            mask[
                height - border:,
                :
            ]
        ),
        (
            image[:, :border],
            mask[:, :border]
        ),
        (
            image[
                :,
                width - border:
            ],
            mask[
                :,
                width - border:
            ]
        )
    ]

    scores = []

    for region, region_mask in regions:

        score = calculate_white_percentage(
            region,
            region_mask
        )

        if score is not None:
            scores.append(score)

    if not scores:

        return {
            "score": 0.0,
            "passed": False
        }

    final_score = float(
        np.mean(scores)
    )

    passed = (
        final_score >=
        BACKGROUND_PASS_SCORE
    )

    return {
        "score": final_score,
        "passed": passed
    }


# ============================================================
# POSITION ANALYSIS
# ============================================================

def analyze_position(
    image,
    face
):

    height, width = image.shape[:2]

    x = float(face[0])
    y = float(face[1])
    w = float(face[2])
    h = float(face[3])

    center_x = x + (w / 2)
    center_y = y + (h / 2)

    image_center_x = width / 2

    horizontal_offset = (
        center_x -
        image_center_x
    ) / width

    vertical_position = (
        center_y / height
    )

    face_width_ratio = (
        w / width
    )

    top_margin_ratio = (
        y / height
    )

    horizontal_ok = (
        abs(horizontal_offset)
        <= MAX_HORIZONTAL_OFFSET
    )

    size_ok = (
        MIN_FACE_WIDTH_RATIO
        <= face_width_ratio
        <= MAX_FACE_WIDTH_RATIO
    )

    top_margin_ok = (
        MIN_TOP_MARGIN_RATIO
        <= top_margin_ratio
        <= MAX_TOP_MARGIN_RATIO
    )

    vertical_ok = (
        0.20
        <= vertical_position
        <= 0.65
    )

    passed = (
        horizontal_ok
        and size_ok
        and top_margin_ok
        and vertical_ok
    )

    return {

        "passed": passed,

        "horizontal_ok":
            horizontal_ok,

        "size_ok":
            size_ok,

        "top_margin_ok":
            top_margin_ok,

        "vertical_ok":
            vertical_ok,

        "horizontal_offset":
            horizontal_offset,

        "vertical_position":
            vertical_position,

        "face_width_ratio":
            face_width_ratio,

        "top_margin_ratio":
            top_margin_ratio
    }


# ============================================================
# MAIN ANALYZER
# ============================================================

def analyze_image(image):

    detections = detect_faces(
        image
    )

    face_count = len(
        detections
    )

    one_person = (
        face_count == 1
    )

    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

    if one_person:

        position = analyze_position(
            image,
            detections[0]
        )

    else:

        position = {
            "passed": False,
            "horizontal_ok": False,
            "size_ok": False,
            "top_margin_ok": False,
            "vertical_ok": False,
            "horizontal_offset": 0,
            "vertical_position": 0,
            "face_width_ratio": 0,
            "top_margin_ratio": 0
        }

    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    background = analyze_background(
        image,
        detections
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    overall_pass = (
        one_person
        and position["passed"]
        and background["passed"]
    )

    return {

        "faces_detected":
            face_count,

        "person":
            "PASS"
            if one_person
            else "FAIL",

        "position":
            "PASS"
            if position["passed"]
            else "FAIL",

        "background":
            "PASS"
            if background["passed"]
            else "FAIL",

        "background_score":
            round(
                background["score"],
                2
            ),

        "horizontal_position":
            "PASS"
            if position["horizontal_ok"]
            else "FAIL",

        "face_size":
            "PASS"
            if position["size_ok"]
            else "FAIL",

        "top_margin":
            "PASS"
            if position["top_margin_ok"]
            else "FAIL",

        "vertical_position":
            "PASS"
            if position["vertical_ok"]
            else "FAIL",

        "overall":
            "PASS"
            if overall_pass
            else "FAIL"
    }


# ============================================================
# WEB API
# ============================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "service": "Student Photo Analyzer"
    }


@app.post("/analyze")
async def analyze_photo(
    file: UploadFile = File(...)
):

    try:

        image_bytes = await file.read()

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return JSONResponse(
                status_code=400,
                content={
                    "error":
                        "Could not read image."
                }
            )

        result = analyze_image(
            image
        )

        result["filename"] = (
            file.filename
        )

        return result

    except Exception as error:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(error)
            }
        )