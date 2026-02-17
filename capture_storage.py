import os
import cv2
import glob
from uuid import uuid4

# --------------------------
# 4. CAPTURE STORAGE (Debug on Disk)
# --------------------------

CAPTURE_DIR = "captures"
MAX_CAPTURES = 200

# Ensure directory exists
if not os.path.exists(CAPTURE_DIR):
    os.makedirs(CAPTURE_DIR)

def get_captures():
    """Returns a list of image filenames currently in the capture directory."""
    files = sorted(glob.glob(os.path.join(CAPTURE_DIR, "*.jpg")), key=os.path.getmtime, reverse=True)
    return {"count": len(files), "captures": [os.path.basename(f) for f in files]}

def cleanup_old_captures():
    """Keeps only the latest MAX_CAPTURES images."""
    files = sorted(glob.glob(os.path.join(CAPTURE_DIR, "*.jpg")), key=os.path.getmtime)
    while len(files) > MAX_CAPTURES:
        try:
            os.remove(files[0])
            files.pop(0)
        except OSError:
            pass

def save_to_history(frame, scan_id, scan_type):
    """Saves a frame as a JPEG file to disk."""
    try:
        cleanup_old_captures()
        
        # Format: timestamp_type_scanid.jpg
        filename = f"{uuid4().hex[:8]}_{scan_type}_{scan_id[:8]}.jpg"
        filepath = os.path.join(CAPTURE_DIR, filename)
        
        # Save directly
        cv2.imwrite(filepath, frame)
        print(f"📸 Saved debug capture: {filepath}")
        
    except Exception as e:
        print(f"⚠️ Failed to save capture to disk: {e}")
