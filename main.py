from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import tempfile


# ============================================================
# STUDENT PHOTO ANALYZER
# ============================================================

app = FastAPI(
    title="Student Photo Analyzer",
    description="Automatic student photo validation API",
    version="8.0"
)


# ============================================================
# SETTINGS
# ============================================================

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
# MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_detection_yunet_2026may.onnx"
)


# ============================================================
# LOAD YUNET
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "YuNet model not found: " + MODEL_PATH
    )


face_detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 320),
    FACE_SCORE_THRESHOLD,
    NMS_THRESHOLD,
    5000
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Student Photo Analyzer",
        "version": "8.0"
    }


# ============================================================
# HELPER: FACE DETECTION
# ============================================================

def detect_faces(image):
    height, width = image.shape[:2]

    face_detector.setInputSize((width, height))

    _, faces = face_detector.detect(image)

    if faces is None:
        return []

    return faces


# ============================================================
# HELPER: BACKGROUND SCORE
# ============================================================

def calculate_background_score(image):
    """
    Calculates the percentage of pixels that look white-ish.

    White-ish means:
    brightness >= 190
    AND
    difference between strongest and weakest RGB channel <= 35
    """

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    brightness = np.mean(rgb, axis=2)

    color_difference = (
        np.max(rgb, axis=2) -
        np.min(rgb, axis=2)
    )

    white_mask = (
        (brightness >= WHITE_BRIGHTNESS) &
        (color_difference <= WHITE_COLOR_DIFFERENCE)
    )

    score = (
        np.sum(white_mask) /
        white_mask.size
    ) * 100.0

    return round(float(score), 2)


# ============================================================
# HELPER: CHECK FACE POSITION
# ============================================================

def check_horizontal_position(
    face_x,
    face_width,
    image_width
):
    face_center_x = face_x + (face_width / 2)

    image_center_x = image_width / 2

    offset = abs(
        face_center_x - image_center_x
    ) / image_width

    passed = offset <= MAX_HORIZONTAL_OFFSET

    return passed, offset


# ============================================================
# HELPER: CHECK FACE SIZE
# ============================================================

def check_face_size(
    face_width,
    image_width
):
    ratio = face_width / image_width

    passed = (
        MIN_FACE_WIDTH_RATIO
        <= ratio
        <= MAX_FACE_WIDTH_RATIO
    )

    return passed, ratio


# ============================================================
# HELPER: CHECK TOP MARGIN
# ============================================================

def check_top_margin(
    face_y,
    image_height
):
    ratio = face_y / image_height

    passed = (
        MIN_TOP_MARGIN_RATIO
        <= ratio
        <= MAX_TOP_MARGIN_RATIO
    )

    return passed, ratio


# ============================================================
# HELPER: CHECK VERTICAL POSITION
# ============================================================

def check_vertical_position(
    face_y,
    face_height,
    image_height
):
    face_center_y = face_y + (face_height / 2)

    image_center_y = image_height / 2

    offset = abs(
        face_center_y - image_center_y
    ) / image_height

    # Allow a reasonable vertical range.
    # The top-margin check separately prevents
    # the face from being too close to the top.
    passed = offset <= 0.20

    return passed, offset


# ============================================================
# ANALYZE IMAGE
# ============================================================

def analyze_image(image, filename):
    height, width = image.shape[:2]

    problems = []

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    faces = detect_faces(image)

    faces_detected = len(faces)

    if faces_detected == 1:
        person = "PASS"
    else:
        person = "FAIL"

        if faces_detected == 0:
            problems.append({
                "check": "Person",
                "message": "No face was detected in the photo."
            })

        else:
            problems.append({
                "check": "Person",
                "message": (
                    f"{faces_detected} faces were detected. "
                    "Exactly one person must be visible."
                )
            })


    # Default values
    horizontal_position = "FAIL"
    face_size = "FAIL"
    top_margin = "FAIL"
    vertical_position = "FAIL"

    horizontal_offset = None
    face_width_ratio = None
    top_margin_ratio = None
    vertical_offset = None


    # --------------------------------------------------------
    # FACE-BASED CHECKS
    # --------------------------------------------------------

    if faces_detected == 1:

        face = faces[0]

        face_x = float(face[0])
        face_y = float(face[1])
        face_width = float(face[2])
        face_height = float(face[3])


        # ----------------------------------------------------
        # HORIZONTAL POSITION
        # ----------------------------------------------------

        horizontal_passed, horizontal_offset = (
            check_horizontal_position(
                face_x,
                face_width,
                width
            )
        )

        horizontal_position = (
            "PASS"
            if horizontal_passed
            else "FAIL"
        )

        if not horizontal_passed:

            direction = (
                "left"
                if (
                    face_x + face_width / 2
                ) < width / 2
                else "right"
            )

            problems.append({
                "check": "Horizontal Position",
                "message": (
                    "Face is too far to the "
                    f"{direction}. "
                    "Please center your face horizontally."
                )
            })


        # ----------------------------------------------------
        # FACE SIZE
        # ----------------------------------------------------

        size_passed, face_width_ratio = (
            check_face_size(
                face_width,
                width
            )
        )

        face_size = (
            "PASS"
            if size_passed
            else "FAIL"
        )

        if not size_passed:

            if face_width_ratio < MIN_FACE_WIDTH_RATIO:
                problems.append({
                    "check": "Face Size",
                    "message": (
                        "Face is too small. "
                        "Move closer to the camera."
                    )
                })

            else:
                problems.append({
                    "check": "Face Size",
                    "message": (
                        "Face is too large. "
                        "Move slightly farther from the camera."
                    )
                })


        # ----------------------------------------------------
        # TOP MARGIN
        # ----------------------------------------------------

        top_passed, top_margin_ratio = (
            check_top_margin(
                face_y,
                height
            )
        )

        top_margin = (
            "PASS"
            if top_passed
            else "FAIL"
        )

        if not top_passed:

            if top_margin_ratio < MIN_TOP_MARGIN_RATIO:

                problems.append({
                    "check": "Top Margin",
                    "message": (
                        "There is not enough space above "
                        "the head. Move the camera/photo "
                        "framing slightly upward."
                    )
                })

            else:

                problems.append({
                    "check": "Top Margin",
                    "message": (
                        "There is too much empty space "
                        "above the head. Adjust the framing."
                    )
                })


        # ----------------------------------------------------
        # VERTICAL POSITION
        # ----------------------------------------------------

        vertical_passed, vertical_offset = (
            check_vertical_position(
                face_y,
                face_height,
                height
            )
        )

        vertical_position = (
            "PASS"
            if vertical_passed
            else "FAIL"
        )

        if not vertical_passed:

            direction = (
                "up"
                if (
                    face_y + face_height / 2
                ) < height / 2
                else "down"
            )

            problems.append({
                "check": "Vertical Position",
                "message": (
                    "Face is positioned too far "
                    f"{direction}. Please adjust the "
                    "vertical framing."
                )
            })


    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    background_score = calculate_background_score(image)

    background = (
        "PASS"
        if background_score >= BACKGROUND_PASS_SCORE
        else "FAIL"
    )

    if background == "FAIL":

        problems.append({
            "check": "Background",
            "message": (
                f"Background is not white enough. "
                f"Detected {background_score:.2f}% white-ish "
                f"background, but at least "
                f"{BACKGROUND_PASS_SCORE:.0f}% is required."
            )
        })


    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    overall = (
        "PASS"
        if (
            faces_detected == 1
            and horizontal_position == "PASS"
            and face_size == "PASS"
            and top_margin == "PASS"
            and vertical_position == "PASS"
            and background == "PASS"
        )
        else "FAIL"
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {
        "faces_detected": faces_detected,

        "person": person,

        "horizontal_position":
            horizontal_position,

        "face_size":
            face_size,

        "top_margin":
            top_margin,

        "vertical_position":
            vertical_position,

        "background":
            background,

        "background_score":
            background_score,

        "overall":
            overall,

        "filename":
            filename,

        "problems":
            problems
    }


    return result


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    temp_path = None

    try:

        # ----------------------------------------------------
        # READ FILE
        # ----------------------------------------------------

        contents = await file.read()

        if not contents:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Uploaded file is empty."
                }
            )


        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        image_array = np.frombuffer(
            contents,
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
                    "error": (
                        "The uploaded file could not "
                        "be read as an image."
                    )
                }
            )


        # ----------------------------------------------------
        # ANALYZE
        # ----------------------------------------------------

        filename = file.filename or "unknown"

        result = analyze_image(
            image,
            filename
        )


        return JSONResponse(
            status_code=200,
            content=result
        )


    except Exception as error:

        print(
            "ANALYZER ERROR:",
            repr(error)
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": str(error)
            }
        )


    finally:

        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)

            except Exception:
                pass
