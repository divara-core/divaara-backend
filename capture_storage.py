from collections import deque
import cv2
import base64
from uuid import uuid4
import numpy as np

# --------------------------
# 4. CAPTURE STORAGE (Debug/History)
# --------------------------
CAPTURE_HISTORY = deque(maxlen=200)

def get_captures():
    """Returns the last 200 captured frames (base64 encoded)."""
    return {"count": len(CAPTURE_HISTORY), "captures": list(CAPTURE_HISTORY)}

def save_to_history(frame, scan_id, scan_type):
    """Encodes and saves a frame to the in-memory history."""
    try:
        # Resize to reduce memory usage (optional, but good practice)
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))
            
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        b64_img = base64.b64encode(buffer).decode('utf-8')
        
        CAPTURE_HISTORY.append({
            "scan_id": scan_id,
            "type": scan_type,
            "timestamp": str(uuid4()), # simple unique string/timestamp placeholder
            "image": f"data:image/jpeg;base64,{b64_img}"
        })
    except Exception as e:
        print(f"⚠️ Failed to save capture to history: {e}")
