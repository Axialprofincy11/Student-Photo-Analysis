import math
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response


# ============================================================
# SETTINGS  (all thresholds live here - tune these, not the code)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "YUNET_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "face_detection_yunet_2026may.onnx"),
)

# --- image size -------------------------------------------------
MIN_IMAGE_SIDE = 400          # reject images smaller than this (px, shorter side)
MAX_IMAGE_SIDE = 1280         # larger images are downscaled so thresholds stay consistent
GRABCUT_MAX_SIDE = 400        # person segmentation runs on a small copy for speed

# --- face detection --------------------------------------------
FACE_SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
MIN_SECONDARY_FACE_AREA_RATIO = 0.35   # faces smaller than 35% of the biggest face are ignored
                                       # (posters, photos on the wall, reflections)

# --- background (LAB colour space, OpenCV 8-bit: L 0-255, a/b centred on 128)
BG_WHITE_MIN_L = 170          # lightness. 255 = pure white, ~170 = light grey wall in shade
BG_WHITE_MAX_CHROMA = 18      # colour tint allowed (0 = perfectly neutral)
BACKGROUND_PASS_SCORE = 70.0  # % of background pixels that must be "white"
MIN_BG_PIXEL_RATIO = 0.05     # need at least 5% of the image as visible background
BG_UNIFORMITY_WARN_STD = 30   # only a warning, not a failure
PERSON_MARGIN_RATIO = 0.025   # safety margin around the person (hair/edge halo is ignored)

# --- position --------------------------------------------------
MAX_HORIZONTAL_OFFSET = 0.12
MIN_FACE_WIDTH_RATIO = 0.15
MAX_FACE_WIDTH_RATIO = 0.50
MIN_TOP_MARGIN_RATIO = 0.08
MAX_TOP_MARGIN_RATIO = 0.40
MIN_VERTICAL_POSITION = 0.20
MAX_VERTICAL_POSITION = 0.65

# --- pose (from YuNet's 5 landmarks) ---------------------------
MAX_ROLL_DEGREES = 10         # head tilt
MAX_YAW_RATIO = 0.25          # nose offset from eye-centre / eye distance (head turn)

# --- quality ---------------------------------------------------
MIN_SHARPNESS = 60.0          # Laplacian variance of the face crop
MIN_FACE_BRIGHTNESS = 60      # L channel of face crop
MAX_FACE_BRIGHTNESS = 230


# ============================================================
# MODEL LOADING (lazy, with a clear error)
# ============================================================

_detector = None


def get_detector():
    global _detector

    if _detector is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"YuNet model not found: {MODEL_PATH}")

        _detector = cv2.FaceDetectorYN_create(
            MODEL_PATH, "", (320, 320),
            FACE_SCORE_THRESHOLD, NMS_THRESHOLD, 5000,
        )

    return _detector


@asynccontextmanager
async def lifespan(app):
    get_detector()  # fail fast at startup if the model is missing
    yield


app = FastAPI(title="Student Photo Analyzer", lifespan=lifespan)


# ============================================================
# HELPERS
# ============================================================

def resize_for_analysis(image):
    h, w = image.shape[:2]
    longest = max(h, w)

    if longest <= MAX_IMAGE_SIDE:
        return image

    scale = MAX_IMAGE_SIDE / longest
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def clamp(value, low, high):
    return max(low, min(high, value))


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(image):
    """Returns (faces sorted biggest-first, number of tiny faces ignored)."""

    detector = get_detector()
    h, w = image.shape[:2]
    detector.setInputSize((w, h))

    _, detections = detector.detect(image)

    if detections is None or len(detections) == 0:
        return [], 0

    faces = sorted(detections, key=lambda f: float(f[2]) * float(f[3]), reverse=True)

    biggest_area = float(faces[0][2]) * float(faces[0][3])
    kept, ignored = [], 0

    for face in faces:
        area = float(face[2]) * float(face[3])
        if area >= biggest_area * MIN_SECONDARY_FACE_AREA_RATIO:
            kept.append(face)
        else:
            ignored += 1

    return kept, ignored


# ============================================================
# PERSON SEGMENTATION  (so we only judge the REAL background)
# ============================================================

def build_person_mask(image, face):
    """
    Returns a full-size boolean mask: True = person (head, hair, shoulders).
    Uses GrabCut seeded from the detected face. Falls back to the
    geometric guess if GrabCut fails.
    """

    h, w = image.shape[:2]

    scale = min(1.0, GRABCUT_MAX_SIDE / max(h, w))
    small = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0 else image.copy()
    )
    sh, sw = small.shape[:2]

    fx, fy, fw, fh = [float(v) * scale for v in face[:4]]

    def px(v):
        return int(clamp(v, 0, sw - 1))

    def py(v):
        return int(clamp(v, 0, sh - 1))

    mask = np.full((sh, sw), cv2.GC_PR_BGD, dtype=np.uint8)

    # probable person: head area
    cv2.rectangle(
        mask,
        (px(fx - 0.5 * fw), py(fy - 0.6 * fh)),
        (px(fx + 1.5 * fw), py(fy + 1.1 * fh)),
        cv2.GC_PR_FGD, -1,
    )

    # probable person: shoulders / torso widening downwards
    body = np.array([
        [fx - 0.2 * fw, fy + fh],
        [fx + 1.2 * fw, fy + fh],
        [fx + 2.2 * fw, sh],
        [fx - 1.2 * fw, sh],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [body], cv2.GC_PR_FGD)

    # sure background: thin strips on left / right / top edges
    edge = max(2, int(0.03 * min(sh, sw)))
    mask[:, :edge] = cv2.GC_BGD
    mask[:, sw - edge:] = cv2.GC_BGD
    mask[:edge, :] = cv2.GC_BGD

    # sure person: core of the face
    mask[
        py(fy + 0.2 * fh): py(fy + 0.8 * fh) + 1,
        px(fx + 0.2 * fw): px(fx + 0.8 * fw) + 1,
    ] = cv2.GC_FGD

    try:
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
    except Exception:
        pass  # keep the geometric seed mask

    person_small = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    person = cv2.resize(person_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # grow the person a little so hair edges / blur halos don't count as background
    k = max(3, int(min(h, w) * PERSON_MARGIN_RATIO) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    person = cv2.dilate(person, kernel)

    return person.astype(bool)


# ============================================================
# BACKGROUND ANALYSIS
# ============================================================

def analyze_background(image, face):
    empty = {
        "score": 0.0, "passed": False, "bg_pixel_ratio": 0.0,
        "median_lightness": 0.0, "mean_chroma": 0.0, "lightness_std": 0.0,
        "reasons": [], "warnings": [], "person_mask": None, "white_mask": None,
    }

    if face is None:
        empty["reasons"].append("Background not checked: no face found to locate the person.")
        return empty

    h, w = image.shape[:2]

    person = build_person_mask(image, face)
    bg_mask = ~person

    bg_ratio = float(bg_mask.sum()) / (h * w)
    empty["person_mask"] = person
    empty["bg_pixel_ratio"] = bg_ratio

    if bg_ratio < MIN_BG_PIXEL_RATIO:
        empty["reasons"].append(
            "Not enough background visible (person fills almost the whole frame)."
        )
        return empty

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)

    white_mask = (L >= BG_WHITE_MIN_L) & (chroma <= BG_WHITE_MAX_CHROMA)

    bg_L = L[bg_mask]
    bg_chroma = chroma[bg_mask]
    white_pct = float(white_mask[bg_mask].mean() * 100.0)

    median_L = float(np.median(bg_L))
    mean_chroma = float(bg_chroma.mean())
    L_std = float(bg_L.std())

    passed = white_pct >= BACKGROUND_PASS_SCORE

    reasons, warnings = [], []

    if not passed:
        if median_L < BG_WHITE_MIN_L:
            reasons.append(
                f"Background is too dark/grey (brightness {median_L:.0f}, need >= {BG_WHITE_MIN_L})."
            )
        if mean_chroma > BG_WHITE_MAX_CHROMA:
            reasons.append(
                f"Background has a colour tint (tint {mean_chroma:.0f}, allowed <= {BG_WHITE_MAX_CHROMA})."
            )
        if not reasons:
            reasons.append(
                f"Only {white_pct:.0f}% of the background is white (need >= {BACKGROUND_PASS_SCORE:.0f}%). "
                "Likely shadows or uneven lighting."
            )

    if L_std > BG_UNIFORMITY_WARN_STD:
        warnings.append("Background is uneven (shadows, objects or gradient).")

    return {
        "score": white_pct, "passed": passed, "bg_pixel_ratio": bg_ratio,
        "median_lightness": median_L, "mean_chroma": mean_chroma,
        "lightness_std": L_std, "reasons": reasons, "warnings": warnings,
        "person_mask": person, "white_mask": white_mask,
    }


# ============================================================
# POSITION ANALYSIS
# ============================================================

def analyze_position(image, face):
    height, width = image.shape[:2]

    x, y, w, h = [float(v) for v in face[:4]]

    center_x = x + w / 2
    center_y = y + h / 2

    horizontal_offset = (center_x - width / 2) / width
    vertical_position = center_y / height
    face_width_ratio = w / width
    top_margin_ratio = y / height

    horizontal_ok = abs(horizontal_offset) <= MAX_HORIZONTAL_OFFSET
    size_ok = MIN_FACE_WIDTH_RATIO <= face_width_ratio <= MAX_FACE_WIDTH_RATIO
    top_margin_ok = MIN_TOP_MARGIN_RATIO <= top_margin_ratio <= MAX_TOP_MARGIN_RATIO
    vertical_ok = MIN_VERTICAL_POSITION <= vertical_position <= MAX_VERTICAL_POSITION

    reasons = []

    if not horizontal_ok:
        side = "right" if horizontal_offset > 0 else "left"
        reasons.append(f"Face is off-centre (shifted {abs(horizontal_offset) * 100:.0f}% to the {side}).")

    if not size_ok:
        if face_width_ratio < MIN_FACE_WIDTH_RATIO:
            reasons.append("Face is too small - move closer / crop tighter.")
        else:
            reasons.append("Face is too large - move back / leave more space.")

    if not top_margin_ok:
        if top_margin_ratio < MIN_TOP_MARGIN_RATIO:
            reasons.append("Not enough space above the head.")
        else:
            reasons.append("Too much space above the head.")

    if not vertical_ok:
        reasons.append("Face is too high or too low in the frame.")

    return {
        "passed": horizontal_ok and size_ok and top_margin_ok and vertical_ok,
        "horizontal_ok": horizontal_ok,
        "size_ok": size_ok,
        "top_margin_ok": top_margin_ok,
        "vertical_ok": vertical_ok,
        "horizontal_offset": horizontal_offset,
        "vertical_position": vertical_position,
        "face_width_ratio": face_width_ratio,
        "top_margin_ratio": top_margin_ratio,
        "reasons": reasons,
    }


# ============================================================
# POSE ANALYSIS  (tilt / turned head, from landmarks)
# ============================================================

def analyze_pose(face):
    # YuNet landmarks: [4,5] right eye, [6,7] left eye, [8,9] nose tip
    eyes = sorted(
        [(float(face[4]), float(face[5])), (float(face[6]), float(face[7]))],
        key=lambda p: p[0],
    )
    (x1, y1), (x2, y2) = eyes
    nose_x = float(face[8])

    eye_distance = math.hypot(x2 - x1, y2 - y1)

    if eye_distance < 1:
        return {"passed": False, "roll_degrees": 0.0, "yaw_ratio": 0.0,
                "reasons": ["Could not measure head pose."]}

    roll = math.degrees(math.atan2(y2 - y1, x2 - x1))
    yaw = (nose_x - (x1 + x2) / 2) / eye_distance

    reasons = []
    if abs(roll) > MAX_ROLL_DEGREES:
        reasons.append(f"Head is tilted ({abs(roll):.0f} deg, max {MAX_ROLL_DEGREES}).")
    if abs(yaw) > MAX_YAW_RATIO:
        reasons.append("Head is turned - look straight at the camera.")

    return {"passed": not reasons, "roll_degrees": roll, "yaw_ratio": yaw, "reasons": reasons}


# ============================================================
# IMAGE QUALITY  (blur + face exposure)
# ============================================================

def analyze_quality(image, face):
    h, w = image.shape[:2]

    x1 = int(clamp(face[0], 0, w - 1))
    y1 = int(clamp(face[1], 0, h - 1))
    x2 = int(clamp(face[0] + face[2], 1, w))
    y2 = int(clamp(face[1] + face[3], 1, h))

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return {"passed": False, "sharpness": 0.0, "face_brightness": 0.0,
                "reasons": ["Could not read face region."]}

    # normalise crop size so sharpness doesn't depend on photo resolution
    crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[:, :, 0].mean())

    reasons = []
    if sharpness < MIN_SHARPNESS:
        reasons.append("Photo is blurry.")
    if brightness < MIN_FACE_BRIGHTNESS:
        reasons.append("Face is too dark.")
    if brightness > MAX_FACE_BRIGHTNESS:
        reasons.append("Face is overexposed.")

    return {"passed": not reasons, "sharpness": sharpness,
            "face_brightness": brightness, "reasons": reasons}


# ============================================================
# MAIN ANALYZER
# ============================================================

def run_analysis(original_image):
    orig_h, orig_w = original_image.shape[:2]
    resolution_ok = min(orig_h, orig_w) >= MIN_IMAGE_SIDE

    image = resize_for_analysis(original_image)

    faces, ignored_faces = detect_faces(image)
    face_count = len(faces)
    one_person = face_count == 1
    main_face = faces[0] if face_count >= 1 else None

    reasons = []

    if not resolution_ok:
        reasons.append(f"Image resolution too low (shorter side {min(orig_h, orig_w)}px, need >= {MIN_IMAGE_SIDE}px).")

    if face_count == 0:
        reasons.append("No face detected.")
    elif face_count > 1:
        reasons.append(f"{face_count} faces detected - only one person allowed.")

    # ---- position / pose / quality (need exactly one face) ----
    if one_person:
        position = analyze_position(image, main_face)
        pose = analyze_pose(main_face)
        quality = analyze_quality(image, main_face)
    else:
        position = {
            "passed": False, "horizontal_ok": False, "size_ok": False,
            "top_margin_ok": False, "vertical_ok": False,
            "horizontal_offset": 0, "vertical_position": 0,
            "face_width_ratio": 0, "top_margin_ratio": 0, "reasons": [],
        }
        pose = {"passed": False, "roll_degrees": 0.0, "yaw_ratio": 0.0, "reasons": []}
        quality = {"passed": False, "sharpness": 0.0, "face_brightness": 0.0, "reasons": []}

    # ---- background (uses largest face to locate the person) ----
    background = analyze_background(image, main_face)

    reasons += position["reasons"] + pose["reasons"] + quality["reasons"] + background["reasons"]

    overall_pass = (
        resolution_ok and one_person and position["passed"]
        and pose["passed"] and quality["passed"] and background["passed"]
    )

    def pf(ok):
        return "PASS" if ok else "FAIL"

    result = {
        "faces_detected": face_count,
        "ignored_small_faces": ignored_faces,

        "resolution": pf(resolution_ok),
        "person": pf(one_person),
        "position": pf(position["passed"]),
        "pose": pf(pose["passed"]),
        "quality": pf(quality["passed"]),
        "background": pf(background["passed"]),
        "background_score": round(background["score"], 2),

        "horizontal_position": pf(position["horizontal_ok"]),
        "face_size": pf(position["size_ok"]),
        "top_margin": pf(position["top_margin_ok"]),
        "vertical_position": pf(position["vertical_ok"]),

        "overall": pf(overall_pass),
        "reasons": reasons,
        "warnings": background["warnings"],

        # raw measured numbers - makes every decision explainable
        "details": {
            "image_size": [orig_w, orig_h],
            "background": {
                "white_percent": round(background["score"], 2),
                "required_percent": BACKGROUND_PASS_SCORE,
                "background_area_percent": round(background["bg_pixel_ratio"] * 100, 1),
                "median_lightness": round(background["median_lightness"], 1),
                "mean_tint": round(background["mean_chroma"], 1),
                "lightness_std": round(background["lightness_std"], 1),
            },
            "position": {
                "horizontal_offset": round(position["horizontal_offset"], 3),
                "vertical_position": round(position["vertical_position"], 3),
                "face_width_ratio": round(position["face_width_ratio"], 3),
                "top_margin_ratio": round(position["top_margin_ratio"], 3),
            },
            "pose": {
                "roll_degrees": round(pose["roll_degrees"], 1),
                "yaw_ratio": round(pose["yaw_ratio"], 3),
            },
            "quality": {
                "sharpness": round(quality["sharpness"], 1),
                "face_brightness": round(quality["face_brightness"], 1),
            },
        },
    }

    debug = {"image": image, "faces": faces, "background": background}
    return result, debug


def analyze_image(image):
    result, _ = run_analysis(image)
    return result


def render_debug(debug):
    """Green = background counted as white, red = background rejected, yellow = person."""

    img = debug["image"].copy()
    bg = debug["background"]
    overlay = img.copy()

    person = bg.get("person_mask")
    white = bg.get("white_mask")

    if person is not None and white is not None:
        bg_mask = ~person
        overlay[bg_mask & white] = (0, 200, 0)
        overlay[bg_mask & ~white] = (0, 0, 220)
        overlay[person] = (0, 215, 255)
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

    for f in debug["faces"]:
        x, y, w, h = [int(v) for v in f[:4]]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for i in range(5):
            cv2.circle(img, (int(f[4 + 2 * i]), int(f[5 + 2 * i])), 3, (255, 255, 255), -1)

    ok, png = cv2.imencode(".png", img)
    return png.tobytes() if ok else b""


# ============================================================
# WEB API
# ============================================================

async def read_image(file: UploadFile):
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # EXIF rotation is applied by OpenCV


@app.get("/")
def home():
    return {"status": "online", "service": "Student Photo Analyzer"}


@app.post("/analyze")
async def analyze_photo(file: UploadFile = File(...)):
    try:
        image = await read_image(file)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Could not read image."})

        result = analyze_image(image)
        result["filename"] = file.filename
        return result

    except Exception as error:
        return JSONResponse(status_code=500, content={"error": str(error)})


@app.post("/debug")
async def debug_photo(file: UploadFile = File(...)):
    """Returns a PNG showing what the analyzer 'sees'. Use this when a result looks wrong."""
    try:
        image = await read_image(file)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Could not read image."})

        _, debug = run_analysis(image)
        return Response(content=render_debug(debug), media_type="image/png")

    except Exception as error:
        return JSONResponse(status_code=500, content={"error": str(error)})
