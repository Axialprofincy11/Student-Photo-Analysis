from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import threading


# ============================================================
# STUDENT PHOTO ANALYZER
# ============================================================

app = FastAPI(
    title="Student Photo Analyzer",
    description="Automatic student photo validation API",
    version="9.1-test"
)


# ============================================================
# SETTINGS
# ============================================================

FACE_SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3

BACKGROUND_PASS_SCORE = 33.0

WHITE_BRIGHTNESS = 190
WHITE_COLOR_DIFFERENCE = 35

MAX_HORIZONTAL_OFFSET = 0.12

MIN_FACE_WIDTH_RATIO = 0.15
MAX_FACE_WIDTH_RATIO = 0.50

MIN_TOP_MARGIN_RATIO = 0.08
MAX_TOP_MARGIN_RATIO = 0.40

MAX_VERTICAL_OFFSET = 0.20

MAX_UPLOAD_SIZE = 15 * 1024 * 1024

MAX_IMAGE_DIMENSION = 1800

BACKGROUND_ANALYSIS_SIZE = 900


# ============================================================
# MODEL
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

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


detector_lock = threading.Lock()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "Student Photo Analyzer",
        "version": "9.1-test"
    }


# ============================================================
# TEMPORARY UPLOAD TEST
# ============================================================
#
# IMPORTANT:
# This endpoint does NOT:
# - decode the image
# - run OpenCV
# - run YuNet
# - calculate background
# - perform any photo validation
#
# It ONLY checks whether the multipart upload from
# Google Apps Script actually reaches FastAPI.
#
# ============================================================

@app.post("/analyze-test")
async def analyze_test(
    file: UploadFile = File(...)
):

    print("")
    print("========================================")
    print("TEMPORARY ANALYZE-TEST REQUEST RECEIVED")
    print("========================================")

    print(
        "Filename:",
        file.filename
    )

    print(
        "Content Type:",
        file.content_type
    )


    try:

        # Read the uploaded file in small chunks.
        # We do this only to determine its size.
        total_size = 0

        while True:

            chunk = await file.read(
                1024 * 1024
            )

            if not chunk:
                break

            total_size += len(chunk)


        print(
            "Upload Size:",
            total_size,
            "bytes"
        )

        print(
            "========================================"
        )


        return JSONResponse(
            status_code=200,
            content={
                "test": "SUCCESS",
                "message":
                    "The upload successfully reached FastAPI.",

                "filename":
                    file.filename,

                "content_type":
                    file.content_type,

                "size_bytes":
                    total_size
            }
        )


    except Exception as error:

        print(
            "ANALYZE-TEST ERROR:",
            repr(error)
        )

        return JSONResponse(
            status_code=200,
            content={
                "test": "FAIL",

                "message":
                    "The request reached FastAPI, "
                    "but reading the uploaded file failed.",

                "error":
                    str(error)
            }
        )


    finally:

        try:
            await file.close()

        except Exception:
            pass


# ============================================================
# HELPER: CLEAN FAIL RESULT
# ============================================================

def fail_result(
    filename,
    problems
):

    return {

        "faces_detected":
            0,

        "person":
            "FAIL",

        "horizontal_position":
            "FAIL",

        "face_size":
            "FAIL",

        "top_margin":
            "FAIL",

        "vertical_position":
            "FAIL",

        "background":
            "FAIL",

        "background_score":
            0.0,

        "overall":
            "FAIL",

        "filename":
            filename,

        "problems":
            problems
    }


# ============================================================
# HELPER: SAFE FACE DETECTION
# ============================================================

def detect_faces(image):

    try:

        height, width = image.shape[:2]

        if (
            width <= 0
            or height <= 0
        ):

            return []


        with detector_lock:

            face_detector.setInputSize(
                (width, height)
            )

            _, faces = (
                face_detector.detect(image)
            )


        if faces is None:

            return []


        return faces


    except Exception as error:

        print(
            "FACE DETECTION ERROR:",
            repr(error)
        )

        return []


# ============================================================
# HELPER: RESIZE IMAGE
# ============================================================

def resize_for_analysis(image):

    height, width = image.shape[:2]

    if (
        height <= MAX_IMAGE_DIMENSION
        and
        width <= MAX_IMAGE_DIMENSION
    ):

        return image


    scale = min(
        MAX_IMAGE_DIMENSION / width,
        MAX_IMAGE_DIMENSION / height
    )


    new_width = max(
        1,
        int(width * scale)
    )


    new_height = max(
        1,
        int(height * scale)
    )


    return cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# HELPER: BACKGROUND SCORE
# ============================================================

def calculate_background_score(image):

    try:

        height, width = image.shape[:2]


        scale = min(
            1.0,
            BACKGROUND_ANALYSIS_SIZE / width,
            BACKGROUND_ANALYSIS_SIZE / height
        )


        if scale < 1.0:

            new_width = max(
                1,
                int(width * scale)
            )


            new_height = max(
                1,
                int(height * scale)
            )


            image = cv2.resize(
                image,
                (
                    new_width,
                    new_height
                ),
                interpolation=cv2.INTER_AREA
            )


        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        brightness = np.mean(
            rgb,
            axis=2
        )


        color_difference = (
            np.max(rgb, axis=2)
            -
            np.min(rgb, axis=2)
        )


        white_mask = (
            (brightness >= WHITE_BRIGHTNESS)
            &
            (
                color_difference
                <= WHITE_COLOR_DIFFERENCE
            )
        )


        score = (
            np.sum(white_mask)
            /
            white_mask.size
        ) * 100.0


        return round(
            float(score),
            2
        )


    except Exception as error:

        print(
            "BACKGROUND ERROR:",
            repr(error)
        )

        return 0.0


# ============================================================
# HELPER: HORIZONTAL POSITION
# ============================================================

def check_horizontal_position(
    face_x,
    face_width,
    image_width
):

    face_center_x = (
        face_x
        +
        face_width / 2
    )


    image_center_x = (
        image_width / 2
    )


    offset = (
        abs(
            face_center_x
            -
            image_center_x
        )
        /
        image_width
    )


    passed = (
        offset
        <=
        MAX_HORIZONTAL_OFFSET
    )


    return passed, offset


# ============================================================
# HELPER: FACE SIZE
# ============================================================

def check_face_size(
    face_width,
    image_width
):

    ratio = (
        face_width
        /
        image_width
    )


    passed = (
        MIN_FACE_WIDTH_RATIO
        <=
        ratio
        <=
        MAX_FACE_WIDTH_RATIO
    )


    return passed, ratio


# ============================================================
# HELPER: TOP MARGIN
# ============================================================

def check_top_margin(
    face_y,
    image_height
):

    ratio = (
        face_y
        /
        image_height
    )


    passed = (
        MIN_TOP_MARGIN_RATIO
        <=
        ratio
        <=
        MAX_TOP_MARGIN_RATIO
    )


    return passed, ratio


# ============================================================
# HELPER: VERTICAL POSITION
# ============================================================

def check_vertical_position(
    face_y,
    face_height,
    image_height
):

    face_center_y = (
        face_y
        +
        face_height / 2
    )


    image_center_y = (
        image_height / 2
    )


    offset = (
        abs(
            face_center_y
            -
            image_center_y
        )
        /
        image_height
    )


    passed = (
        offset
        <=
        MAX_VERTICAL_OFFSET
    )


    return passed, offset


# ============================================================
# ANALYZE IMAGE
# ============================================================

def analyze_image(
    image,
    filename
):

    problems = []


    if image is None:

        return fail_result(
            filename,
            [
                {
                    "check": "Image",
                    "message":
                        "The uploaded image could not be read."
                }
            ]
        )


    height, width = image.shape[:2]


    if (
        width <= 0
        or height <= 0
    ):

        return fail_result(
            filename,
            [
                {
                    "check": "Image",
                    "message":
                        "The uploaded image has invalid dimensions."
                }
            ]
        )


    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    try:

        image = resize_for_analysis(
            image
        )

        height, width = image.shape[:2]


    except Exception as error:

        print(
            "RESIZE ERROR:",
            repr(error)
        )


        return fail_result(
            filename,
            [
                {
                    "check": "Image",
                    "message":
                        "The image could not be prepared for analysis."
                }
            ]
        )


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    faces = detect_faces(
        image
    )


    faces_detected = len(
        faces
    )


    if faces_detected == 1:

        person = "PASS"


    else:

        person = "FAIL"


        if faces_detected == 0:

            problems.append({
                "check":
                    "Person",

                "message":
                    "No face was detected in the photo."
            })


        else:

            problems.append({
                "check":
                    "Person",

                "message":
                    f"{faces_detected} faces were detected. "
                    "Exactly one person must be visible."
            })


    # Default values

    horizontal_position = "FAIL"
    face_size = "FAIL"
    top_margin = "FAIL"
    vertical_position = "FAIL"


    # --------------------------------------------------------
    # FACE CHECKS
    # --------------------------------------------------------

    if faces_detected == 1:

        face = faces[0]


        face_x = float(
            face[0]
        )

        face_y = float(
            face[1]
        )

        face_width = float(
            face[2]
        )

        face_height = float(
            face[3]
        )


        # ----------------------------------------------------
        # HORIZONTAL
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
                    face_x
                    +
                    face_width / 2
                )
                <
                width / 2
                else
                "right"
            )


            problems.append({
                "check":
                    "Horizontal Position",

                "message":
                    "Face is too far to the "
                    f"{direction}. "
                    "Please center your face horizontally."
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

            if (
                face_width_ratio
                <
                MIN_FACE_WIDTH_RATIO
            ):

                problems.append({
                    "check":
                        "Face Size",

                    "message":
                        "Face is too small. "
                        "Move closer to the camera."
                })


            else:

                problems.append({
                    "check":
                        "Face Size",

                    "message":
                        "Face is too large. "
                        "Move slightly farther from the camera."
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

            if (
                top_margin_ratio
                <
                MIN_TOP_MARGIN_RATIO
            ):

                problems.append({
                    "check":
                        "Top Margin",

                    "message":
                        "There is not enough space above "
                        "the head. Move the camera/photo "
                        "framing slightly upward."
                })


            else:

                problems.append({
                    "check":
                        "Top Margin",

                    "message":
                        "There is too much empty space "
                        "above the head. Adjust the framing."
                })


        # ----------------------------------------------------
        # VERTICAL
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
                    face_y
                    +
                    face_height / 2
                )
                <
                height / 2
                else
                "down"
            )


            problems.append({
                "check":
                    "Vertical Position",

                "message":
                    "Face is positioned too far "
                    f"{direction}. Please adjust the "
                    "vertical framing."
            })


    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    background_score = (
        calculate_background_score(
            image
        )
    )


    background = (
        "PASS"
        if
        background_score
        >=
        BACKGROUND_PASS_SCORE
        else
        "FAIL"
    )


    if background == "FAIL":

        problems.append({
            "check":
                "Background",

            "message":
                f"Background is not white enough. "
                f"Detected {background_score:.2f}% "
                "white-ish background, but at least "
                f"{BACKGROUND_PASS_SCORE:.0f}% is required."
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
        else
        "FAIL"
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "faces_detected":
            faces_detected,

        "person":
            person,

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


# ============================================================
# REAL ANALYZE ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...)
):

    filename = (
        file.filename
        or
        "unknown"
    )


    try:

        # ----------------------------------------------------
        # READ UPLOAD SAFELY
        # ----------------------------------------------------

        chunks = []

        total_size = 0


        while True:

            chunk = await file.read(
                1024 * 1024
            )


            if not chunk:

                break


            total_size += len(
                chunk
            )


            if (
                total_size
                >
                MAX_UPLOAD_SIZE
            ):

                print(
                    "UPLOAD TOO LARGE:",
                    filename,
                    total_size
                )


                result = fail_result(
                    filename,
                    [
                        {
                            "check":
                                "Image",

                            "message":
                                "The uploaded image file is too large. "
                                "Please upload a smaller photo."
                        }
                    ]
                )


                return JSONResponse(
                    status_code=200,
                    content=result
                )


            chunks.append(
                chunk
            )


        if total_size == 0:

            result = fail_result(
                filename,
                [
                    {
                        "check":
                            "Image",

                        "message":
                            "The uploaded file is empty."
                    }
                ]
            )


            return JSONResponse(
                status_code=200,
                content=result
            )


        contents = b"".join(
            chunks
        )


        chunks.clear()


        # ----------------------------------------------------
        # DECODE
        # ----------------------------------------------------

        image_array = np.frombuffer(
            contents,
            dtype=np.uint8
        )


        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        del image_array
        del contents


        if image is None:

            result = fail_result(
                filename,
                [
                    {
                        "check":
                            "Image",

                        "message":
                            "The uploaded file is not a valid "
                            "readable image."
                    }
                ]
            )


            return JSONResponse(
                status_code=200,
                content=result
            )


        # ----------------------------------------------------
        # ANALYZE
        # ----------------------------------------------------

        result = analyze_image(
            image,
            filename
        )


        del image


        return JSONResponse(
            status_code=200,
            content=result
        )


    except Exception as error:

        print(
            "ANALYZER ERROR:",
            repr(error)
        )


        result = fail_result(
            filename,
            [
                {
                    "check":
                        "Analyzer",

                    "message":
                        "The photo could not be analyzed safely. "
                        "Please upload a different photo."
                }
            ]
        )


        return JSONResponse(
            status_code=200,
            content=result
        )


    finally:

        try:

            await file.close()

        except Exception:

            pass
