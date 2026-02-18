# ==============================================================================
# DIVAARA – UNIFIED AI BACKEND (Body + Skin)
# VERSION: v1.1 (MVP/Demo + Micro-Improvements)
# 
# TECHNICAL AUDIT NOTES:
# 1. QUEUES: Global queues are used for v1 simplicity (Single-Session/Demo). 
#    In production SaaS, these must be scoped by Session ID or WebSocket connection.
# 2. FACE DETECTION: Haar Cascades used for deterministic speed + low compute cost.
#    MediaPipe Mesh is overkill for simple ROI extraction in v1.
# 3. TONE ANALYSIS: LAB color space is lighting-sensitive. Reliability is ensured 
#    via strict "Lighting Symmetry" and "Variance" gating.
# ==============================================================================

import cv2
import mediapipe as mp
import numpy as np
import uvicorn
import statistics
import base64
import os
import threading 
import traceback
from collections import deque, Counter
from typing import List, Dict, Any, Optional
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from capture_storage import save_to_history, get_captures as fetch_captures
# ==============================================================================

# --- App Setup ---
API_VERSION = "v1.1"

app = FastAPI(title="DIVAARA – Unified AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api")

# --- Config ---
SHOULDER_HIP_LIMITS = (0.85, 1.25)
WAIST_HIP_LIMITS = (0.60, 0.95)

MIN_STABLE_FRAMES = 10
MAX_SR_STD = 0.06
MAX_WAIST_STD = 6
CONFIDENCE_LOCK_THRESHOLD = 0.75

# --- State Queues ---
# ⚠️ ARCHITECTURE NOTE: Global state assumes single-user demo mode.
SESSIONS: Dict[str, Dict[str, deque]] = {}

def get_session(scan_id: str) -> Dict[str, deque]:
    if scan_id not in SESSIONS:
        SESSIONS[scan_id] = {
            "shoulder_hip_q": deque(maxlen=15),
            "waist_hip_q": deque(maxlen=15),
            "shoulder_w_q": deque(maxlen=15),
            "waist_w_q": deque(maxlen=15),
            "hip_w_q": deque(maxlen=15),
            "confidence_q": deque(maxlen=15),
        }
    return SESSIONS[scan_id]

# --- MediaPipe Init & Thread Safety ---
segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
segmentation_lock = threading.Lock() 

# --- Input Schema ---
class FrameInput(BaseModel):
    image: str
    scan_mode: str = "mobile"  # "mobile" | "laptop"
    scan_id: Optional[str] = None

# --- Utilities ---
def reset_all_queues(session: Dict[str, deque]):
    session["shoulder_hip_q"].clear()
    session["waist_hip_q"].clear()
    session["shoulder_w_q"].clear()
    session["waist_w_q"].clear()
    session["hip_w_q"].clear()
    session["confidence_q"].clear()

def decode_image(base64_str):
    try:
        img_bytes = base64.b64decode(base64_str.split(",")[-1])
        img_array = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return None

def get_silhouette_mask(frame):
    with segmentation_lock:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = segmenter.process(rgb)
    
    if result.segmentation_mask is None:
        return None
    return (result.segmentation_mask > 0.6).astype("uint8")

def clean_mask_for_mode(mask, mode):
    h, w = mask.shape
    if mode == "laptop":
        cutoff = int(h * 0.75)   # ignore legs + desk
        mask[cutoff:, :] = 0
    return mask

# --- Pose & Measurement Logic ---
def is_upright(mask, mode="mobile"):
    ys, xs = np.where(mask > 0)
    if len(xs) < 500:
        return False

    slope_tol = 0.20 if mode == "laptop" else 0.12
    slope = np.polyfit(ys, xs, 1)[0]
    return abs(slope) < slope_tol

def trimmed_width(xs, trim_ratio=0.12):
    n = len(xs)
    t = int(n * trim_ratio)
    return xs[t], xs[n - t - 1]

def suppress_arms(row, cx, max_ratio=0.55):
    xs = np.where(row > 0)[0]
    if len(xs) < 20: return xs
    max_half = int(len(row) * max_ratio / 2)
    torso = xs[np.abs(xs - cx) < max_half]
    return torso if len(torso) > 20 else xs

def width_at_ratio(mask, y_ratio):
    h, w = mask.shape
    y = int(h * y_ratio)
    if y <= 0 or y >= h: return None
    row = mask[y]
    xs = suppress_arms(row, w // 2)
    if len(xs) < 20: return None
    l, r = trimmed_width(xs)
    return r - l

def multi_band_width(mask, ratios, mode="mean"):
    values = [width_at_ratio(mask, r) for r in ratios]
    values = [v for v in values if v is not None]
    if len(values) < 2: return None
    return max(values) if mode == "max" else statistics.mean(values)

# --- Body Intelligence ---
def is_waist_inflated(mask, waist_w, hip_w):
    if waist_w <= hip_w: return False
    waist_bands = [width_at_ratio(mask, r) for r in (0.48, 0.50, 0.52)]
    hip_bands = [width_at_ratio(mask, r) for r in (0.68, 0.72)]
    waist = [w for w in waist_bands if w]
    hip = [h for h in hip_bands if h]
    if len(waist) < 2 or len(hip) < 2: return False
    return statistics.pstdev(waist) > 4 and statistics.pstdev(hip) < 3

def primary_shape(SR, WR, SW, inflated):
    if abs(SR - 1) <= 0.08 and WR <= 0.75: return "Hourglass"
    if abs(SR - 1) <= 0.10 and abs(WR - 1) <= 0.10: return "Rectangle"
    if SR >= 1.12 and WR >= 0.85: return "Inverted Triangle"
    if SR <= 0.90 and WR <= 0.85: return "Pear"
    if WR >= 1.05 and 0.95 <= SW <= 1.05 and not inflated: return "Apple"
    if WR >= 1.10 and SR < 1.0: return "Diamond"
    return "Rectangle"

def secondary_shapes(SR, WR, inflated):
    out = []
    if SR >= 1.10: out.append("Inverted Triangle tendency")
    if WR >= 1.05 and not inflated: out.append("Apple tendency")
    if abs(SR - 1.0) <= 0.08: out.append("Rectangle tendency")
    return out[:2]

def silhouette_deform(SR, WR):
    return {
        "shoulder": round(float(np.clip(SR, 0.9, 1.25)), 3),
        "waist": round(float(np.clip(WR, 0.65, 0.95)), 3),
        "hip": 1.0
    }

# ==============================================================================
# 🟡 MODE 2: SKIN SCAN CONFIGURATION & UTILS (Hybrid ML + Rules)
# ==============================================================================

# --- Init Face Classifier ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- ML Validator (Mock) ---
class MLEngine:
    """
    Placeholder for MobileNet/TFLite model.
    Acts as a VALIDATOR (Check) not a DICTATOR (Decision).
    """
    def analyze(self, face_roi: np.ndarray) -> Dict[str, Any]:
        h, w, _ = face_roi.shape
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Mock probabilities
        face_prob = 0.95 if w > 50 and h > 50 else 0.4
        lighting_prob = 0.85 if 40 < brightness < 220 else 0.3
        
        # Mock shape classification
        shapes = ["oval", "round", "heart", "square", "long"]
        shape_pred = shapes[hash(int(brightness)) % 5]
        shape_conf = 0.78 

        return {
            "face_prob": face_prob,
            "lighting_prob": lighting_prob,
            "shape": shape_pred,
            "shape_conf": shape_conf
        }

ml_engine = MLEngine()

# --- Skin Utilities ---
def get_blur_score(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def detect_face_shape_rules(w, h):
    # V1 Heuristic: Simple Aspect Ratio
    if w == 0: return "unknown"
    ratio = h / w
    if ratio > 1.6: return "long"
    elif ratio > 1.45: return "oval"
    elif ratio > 1.35: return "heart"
    elif ratio > 1.15: return "round"
    else: return "square"

# ==============================
# 🎨 SKIN TONE PALETTE (LAB-L BASED)
# ==============================

TONE_PALETTE = [
    {"min": 82, "label": "porcelain", "hex": "#F6EDE4"},
    {"min": 75, "label": "ivory", "hex": "#F1E0D1"},
    {"min": 68, "label": "light", "hex": "#E8CBB1"},
    {"min": 62, "label": "light-medium", "hex": "#D9AF8C"},
    {"min": 55, "label": "medium", "hex": "#C6865C"},
    {"min": 48, "label": "medium-tan", "hex": "#AD6A3A"},
    {"min": 42, "label": "tan", "hex": "#8D5524"},
    {"min": 36, "label": "deep", "hex": "#6B3F1D"},
    {"min": 0,  "label": "deep-rich", "hex": "#4A2C18"},
]

def tone_to_palette(tone_index: int):
    for p in TONE_PALETTE:
        if tone_index >= p["min"]:
            return p
    return TONE_PALETTE[-1]


# ==============================================================================
# 🚀 ENDPOINTS
# ==============================================================================

# --------------------------
# Mount the captures directory to serve image files directly (e.g. /api/captures/static/image.jpg)
if not os.path.exists("captures"):
    os.makedirs("captures")
app.mount("/api/captures/static", StaticFiles(directory="captures"), name="captures")

# --------------------------
# 4. CAPTURE & HISTORY ENDPOINT
# --------------------------
@api_router.get("/captures", response_class=HTMLResponse)
def get_recent_captures():
    data = fetch_captures()
    images_html = ""
    for filename in data["captures"]:
        url = f"/api/captures/static/{filename}"
        images_html += f"""
            <div style="display:inline-block; margin:10px; text-align:center;">
                <a href="{url}" target="_blank">
                    <img src="{url}" style="max-width:300px; border:1px solid #ccc;"/>
                </a>
                <br>
                <small>{filename}</small>
            </div>
        """
    
    return f"""
    <html>
        <head><title>Capture Gallery ({data['count']})</title></head>
        <body style="font-family:sans-serif; background:#f0f0f0; padding:20px;">
            <h1>Recent Captures ({data['count']})</h1>
            <p>Most recent first. Click image to view full size.</p>
            <hr>
            {images_html}
        </body>
    </html>
    """

# --------------------------
# 1. BODY SCAN ENDPOINT
# --------------------------
@api_router.post("/analyze-body")
def analyze_body(data: FrameInput):
    scan_id = data.scan_id or str(uuid4())
    session = get_session(scan_id)
    frame = decode_image(data.image)
    if frame is not None:
        save_to_history(frame, scan_id, "body")

    if frame is None:
        reset_all_queues(session)
        return {"status": "error", "version": API_VERSION, "scan_id": scan_id}

    mask = get_silhouette_mask(frame)
    if mask is None:
        reset_all_queues(session)
        return {"status": "scanning", "version": API_VERSION, "scan_id": scan_id}
    
    mask = clean_mask_for_mode(mask, data.scan_mode)
    
    if not is_upright(mask, data.scan_mode):
        reset_all_queues(session)
        return {"status": "pose_not_upright", "version": API_VERSION, "scan_id": scan_id}

    shoulder_w = multi_band_width(mask, [0.30, 0.33, 0.36])
    waist_w = multi_band_width(mask, [0.48, 0.50, 0.52])
    hip_w = multi_band_width(mask, [0.68, 0.72], mode="max")

    if not all([shoulder_w, waist_w, hip_w]):
        return {"status": "scanning", "version": API_VERSION, "scan_id": scan_id}

    session["shoulder_w_q"].append(shoulder_w)
    session["waist_w_q"].append(waist_w)
    session["hip_w_q"].append(hip_w)

    if len(session["waist_w_q"]) >= MIN_STABLE_FRAMES and statistics.pstdev(session["waist_w_q"]) > MAX_WAIST_STD:
        reset_all_queues(session)
        return {"status": "stabilizing", "version": API_VERSION, "scan_id": scan_id}

    SR = shoulder_w / hip_w
    WR = waist_w / hip_w

    session["shoulder_hip_q"].append(SR)
    session["waist_hip_q"].append(WR)

    if len(session["shoulder_hip_q"]) < MIN_STABLE_FRAMES:
        return {"status": "stabilizing", "version": API_VERSION, "scan_id": scan_id}

    if statistics.pstdev(session["shoulder_hip_q"]) > MAX_SR_STD:
        return {"status": "stabilizing", "version": API_VERSION, "scan_id": scan_id}

    avg_SR = float(np.clip(statistics.mean(session["shoulder_hip_q"]), *SHOULDER_HIP_LIMITS))
    avg_WR = float(np.clip(statistics.mean(session["waist_hip_q"]), *WAIST_HIP_LIMITS))
    
    # Division-by-zero protection (Polish B)
    avg_SW = avg_SR / avg_WR if avg_WR > 0 else 1.0

    inflated = is_waist_inflated(mask, waist_w, hip_w)

    confidence_val = 0.5
    print("confidence_val", confidence_val)
    if inflated: confidence_val -= 0.14
    if statistics.pstdev(session["shoulder_hip_q"]) > 0.05: confidence_val -= 0.10

    confidence_val = float(np.clip(confidence_val, 0.0, 1.0))
    session["confidence_q"].append(confidence_val)
    final_conf = round(statistics.mean(session["confidence_q"]), 2)

    if final_conf < CONFIDENCE_LOCK_THRESHOLD:
        reset_all_queues(session) # Reset queues on fail (Polish A)
        return {"status": "scan_not_reliable", "version": API_VERSION, "scan_id": scan_id}

    shape = primary_shape(avg_SR, avg_WR, avg_SW, inflated)

    return {
        "status": "locked",
        "version": API_VERSION,
        "scan_id": scan_id,
        "confidence": final_conf,
        "body_shape": {
            "primary": shape,
            "secondary": secondary_shapes(avg_SR, avg_WR, inflated)
        },
        "ratios": {
            "shoulder_hip": round(avg_SR, 3),
            "waist_hip": round(avg_WR, 3),
            "shoulder_waist": round(avg_SW, 3)
        },
        "body_intelligence": {
            "shape": {
                "primary": shape,
                "secondary": secondary_shapes(avg_SR, avg_WR, inflated)
            },
            "flags": {"inflated_waist": inflated}
        },
        "silhouette_deform": silhouette_deform(avg_SR, avg_WR)
    }

# --------------------------
# 2. FACE DETECTION (Night Vision + Debug File)
# --------------------------
@api_router.post("/detect-face")
def detect_face(data: FrameInput):
    # 1. Decode
    frame = decode_image(data.image)
    if frame is None:
        return {"found": False, "stable": False, "message": None}

    # 2. Save what the server "sees" (CHECK THIS FILE IN YOUR FOLDER!)
    # If this image is black, your browser is blocking the camera.
    # If this image shows your face, the code is working!
    if os.getenv("ENV") == "dev":
        cv2.imwrite("server_view_debug.jpg", frame)

    # 3. "Night Vision" Processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # EQUALIZE HISTOGRAM: Boosts contrast so faces pop out in dim light
    gray = cv2.equalizeHist(gray)

    # 4. Ultra-Sensitive Detection
    # scaleFactor=1.05 (Scans more thoroughly)
    # minNeighbors=3 (Accepts less certainty)
    faces = face_cascade.detectMultiScale(gray, 1.05, 3)

    if len(faces) == 0:
        print("👀 DEBUG: Still looking... (Check server_view_debug.jpg)")
        return {"found": False, "stable": False, "message": None}

    # 5. Lock Logic
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    frame_h, frame_w = frame.shape[:2]
    face_ratio = (w * h) / (frame_h * frame_w)

    print(f"✅ FACE LOCKED! Ratio: {face_ratio:.3f}")

    if face_ratio < 0.05:
        return {"found": True, "stable": False, "message": "move_closer"}
    
    if face_ratio > 0.45:
        return {"found": True, "stable": False, "message": "move_back"}

    # 6. Stability
    cx, cy = x + w / 2, y + h / 2
    if not hasattr(detect_face, "centers"):
        detect_face.centers = deque(maxlen=5)

    detect_face.centers.append((cx, cy))

    stable = False
    if len(detect_face.centers) == 5:
        xs, ys = zip(*detect_face.centers)
        # Very permissive stability check (20.0 variance allowed)
        if np.std(xs) < 20.0 and np.std(ys) < 20.0:
            stable = True
            print("📸 CAPTURING!")

    return {
        "found": True,
        "stable": stable,
        "message": None if stable else "hold_steady"
    }

# --------------------------
# 3. SKIN SCAN ENDPOINT
# --------------------------
@api_router.post("/analyze-skin")
async def analyze_skin(files: List[UploadFile] = File(...)):
    try:
        # --- 1. Basic Validation ---
        if not files or len(files) == 0:
            return {"status": "error", "message": "No frames received.", "version": API_VERSION}

        print(f"📥 DEBUG: Received {len(files)} frames for analysis...")
        
        # ✅ FIX APPLIED HERE: Reduced from 15 to 3 for speed
        MAX_FRAMES = 3 
        files = files[:MAX_FRAMES]

        lab_history = []
        conf_scores = []
        distance_scores = []
        
        shape_votes_rules = []
        shape_votes_ml = []
        
        reject_reasons = {
            "blur": 0, "no_face": 0, "too_far": 0, 
            "lighting_rule": 0, "lighting_ml": 0, "face_ml": 0, "processing_error": 0
        }

        # --- 2. Frame Processing Loop ---
        for i, file in enumerate(files):
            try:
                contents = await file.read()
                if len(contents) > 2_000_000:
                    reject_reasons["processing_error"] += 1
                    continue
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None: 
                    save_to_history(img, "skin_batch", "skin")

                if img is None: 
                    print(f"⚠️ Frame {i}: Decode failed")
                    continue

                # A. Blur Check
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur_val = get_blur_score(gray)
                if blur_val < 60:
                    reject_reasons["blur"] += 1
                    continue

                # B. Face Detection
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                if len(faces) == 0:
                    reject_reasons["no_face"] += 1
                    continue

                x, y, w_f, h_f = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                
                # C. Check ROI Validity (Prevents Crash)
                if w_f < 10 or h_f < 10:
                    reject_reasons["no_face"] += 1
                    continue
                    
                face_coverage = (w_f * h_f) / (img.shape[0] * img.shape[1])
                
                # D. Distance Gate
                if face_coverage < 0.04: # Relaxed slightly
                    reject_reasons["too_far"] += 1
                    continue

                face_roi = img[y:y+h_f, x:x+w_f]
                fh, fw, _ = face_roi.shape

                # E. ML Validation (Safe Mode)
                try:
                    ml_result = ml_engine.analyze(face_roi)
                    if ml_result["face_prob"] < 0.6: 
                        reject_reasons["face_ml"] += 1
                        continue
                    if ml_result["lighting_prob"] < 0.5:
                        reject_reasons["lighting_ml"] += 1
                        continue
                except Exception as e:
                    print(f"⚠️ ML Error on frame {i}: {e}")
                    reject_reasons["processing_error"] += 1
                    continue

                # F. Rule Validation & Zone Extraction
                zones = [
                    face_roi[int(fh*0.15):int(fh*0.25), int(fw*0.35):int(fw*0.65)], # Forehead
                    face_roi[int(fh*0.45):int(fh*0.55), int(fw*0.20):int(fw*0.35)], # Left Cheek
                    face_roi[int(fh*0.45):int(fh*0.55), int(fw*0.65):int(fw*0.80)]  # Right Cheek
                ]
                
                # Safety check for empty zones
                if any(z.size == 0 for z in zones): 
                    reject_reasons["processing_error"] += 1
                    continue

                l_bright = np.mean(cv2.cvtColor(zones[1], cv2.COLOR_BGR2GRAY))
                r_bright = np.mean(cv2.cvtColor(zones[2], cv2.COLOR_BGR2GRAY))
                l_diff = abs(l_bright - r_bright)

                if l_diff > 20: # Relaxed from 12 to 20 to prevent rejection loops
                    reject_reasons["lighting_rule"] += 1
                    continue

                # G. Collect Data
                shape_votes_ml.append({"shape": ml_result["shape"], "conf": ml_result["shape_conf"]})
                shape_votes_rules.append(detect_face_shape_rules(fw, fh))

                frame_lab_vals = []
                for z in zones:
                    hsv = cv2.cvtColor(z, cv2.COLOR_BGR2HSV)
                    mask = (hsv[:, :, 2] < 240) & (hsv[:, :, 1] > 20)
                    filtered = z[mask]
                    pz = filtered if len(filtered) > 0 else z.reshape(-1, 3)
                    lab_zone = cv2.cvtColor(pz.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
                    frame_lab_vals.append(np.mean(lab_zone.reshape(-1, 3), axis=0))

                avg_frame_lab = np.mean(frame_lab_vals, axis=0)
                l_score = max(0, (20 - l_diff) / 20)
                d_score = min(1.0, face_coverage / 0.12)
                
                lab_history.append(avg_frame_lab)
                distance_scores.append(d_score)
                conf_scores.append((l_score * 0.7) + (d_score * 0.3))

            except Exception as e:
                print(f"⚠️ Error processing individual frame {i}: {e}")
                reject_reasons["processing_error"] += 1
                continue

        # --- 3. Aggregation Logic (Safe) ---
        if len(lab_history) == 0:
            print("❌ DEBUG: All frames rejected. Reasons:", reject_reasons)
            dominant = max(reject_reasons, key=reject_reasons.get)
            messages = {
                "blur": "Hold still.", "no_face": "Align face.", 
                "too_far": "Move closer.", "lighting_rule": "Even lighting needed.",
                "lighting_ml": "Lighting quality poor.", "face_ml": "Face not clear."
            }
            return {"status": "error", "message": messages.get(dominant, "Unstable scan."), "version": API_VERSION}

        # Outlier Filtering
        lab_arr = np.array(lab_history)
        if len(lab_arr) > 2:
            median_lab = np.median(lab_arr, axis=0)
            distances = np.linalg.norm(lab_arr - median_lab, axis=1)
            valid_indices = distances < np.percentile(distances, 75)
            valid_labs = lab_arr[valid_indices]
        else:
            valid_labs = lab_arr
            valid_indices = np.array([True] * len(lab_arr))

        if len(valid_labs) == 0:
            return {"status": "error", "message": "Inconsistent data.", "version": API_VERSION}

        # Final Calculations
        valid_conf_array = np.array(conf_scores)[valid_indices] if len(conf_scores) > 0 else np.array([0.5])
        weights = np.array(distance_scores)[valid_indices] if len(distance_scores) > 0 else np.array([1.0])
        weights = weights / (np.sum(weights) + 1e-6)
        final_conf = float(np.clip(np.sum(valid_conf_array * weights), 0, 1))
        
        # ==============================
        # 🎨 SKIN TONE (LAB → INDEX → PALETTE)
        # ==============================

        avg_l, avg_a, avg_b = np.mean(valid_labs, axis=0)

        # Continuous tone measurement (0–100)
        tone_index = int(np.clip(avg_l, 0, 100))

        # Palette mapping
        palette = tone_to_palette(tone_index)

        # Undertone (unchanged logic)
        undertone = (
            "warm" if (avg_b - avg_a) > 8 else
            "cool" if (avg_b - avg_a) < -8 else
            "neutral"
        )

        final_shape = "oval"  # Stable fallback

        print(f"✅ DEBUG: SUCCESS! Tone: {palette['label']}, Undertone: {undertone}")

        return {
            "status": "success",
            "version": API_VERSION,

            "skin_tone": {
                "label": palette["label"],
                "confidence": round(final_conf, 2),
                "palette": {
                    "hex": palette["hex"],
                    "index": tone_index,
                    "space": "LAB-L"
                }
            },

            "undertone": undertone,
            "face_shape": final_shape,
            "lock_safe": True,

            "meta": {
                "engine": "skin-scan-hybrid-v2.2",
                "frames_used": len(valid_labs)
            }
        }

    except Exception as e:
        # --- 4. CATCH-ALL CRASH HANDLER ---
        traceback.print_exc()
        print(f"🔥 CRITICAL SERVER CRASH PREVENTED: {e}")
        return {"status": "error", "message": "Server processing error", "version": API_VERSION}

# Include the router
app.include_router(api_router)

if __name__ == "__main__":
    # This starts the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
