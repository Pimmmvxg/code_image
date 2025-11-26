# config.py
from pathlib import Path
import re
from typing import Optional

# Debug / I/O configuration
DEBUG_MODE = 'none'      # 'none' | 'print' | 'plot'
THREADS = 1               # Number of threads to use for processing
SAVE_MASK = True          # Save the mask image
SAVE_TOP_OVERLAY = True
SAVE_SIDE_ROIS_OVERVIEW = True  # เซฟรูปภาพรวมกรอบ ROI (#1, #2, ...)
ENABLE_THINGSBOARD = True  # เปิด/ปิดการส่งข้อมูลขึ้น ThingsBoard

# I/O
INPUT_PATH: Optional[Path] = None
#INPUT_PATH: Optional[Path] = Path(r"C:\Cantonese\1\picture_topview_12092025_131834.png")
OUTPUT_DIR: Optional[Path] = None
VIEW: Optional[str] = None       # "side" | "top" | None
EXTENSIONS = ['.png', '.jpg', '.jpeg']  # Supported image file extensions

# Folder-name view patterns
_SIDE_PATTERNS = [r"sideview", r"side[_\- ]?view", r"\bside\b"]
_TOP_PATTERNS  = [r"topview",  r"top[_\- ]?view",  r"\btop\b"]

# External Binary Mask file (optional)
MASK_PATH: Optional[Path] = None   # เช่น r"C:\path\to\my_mask.png"
MASK_SPEC: Optional[dict] = None
USE_EXTERNAL_MASK: bool = False    # True เมื่อ MASK_PATH ถูกตั้งค่า

# Utilities
def _detect_view_from_path(p: Path) -> str:
    s = str(p).lower()
    for pat in _SIDE_PATTERNS:
        if re.search(pat, s):
            return "side"
    for pat in _TOP_PATTERNS:
        if re.search(pat, s):
            return "top"
    return "unknown"

def safe_target_name(p: Optional[Path]) -> str:
    if p is None:
        return "input"
    q = Path(p)
    # ไฟล์: มี suffix หรือ is_file() → ใช้ stem
    if q.suffix or q.is_file():
        return q.stem or "input"
    # โฟลเดอร์ → ใช้ name
    return q.name or "input"

def _default_output_dir(p: Optional[Path]) -> Path:
    if p is None:
        return Path("./results_input")  # fallback กรณีไม่ได้ส่ง path

    path = Path(p)
    try:
        if path.is_file():  
            name = path.stem
        else:
            name = path.name
        if not name:
            name = "input"
    except Exception:
        name = "input"
    return Path(f"./results_{name}")

# -------------------------------------------------
# Main entry to resolve runtime parameters
# -------------------------------------------------
def resolve_runtime(input_path: str | Path,
                    output_dir: Optional[str | Path] = None,
                    view: Optional[str] = None):
   
    global INPUT_PATH, OUTPUT_DIR, VIEW, MASK_SPEC, USE_EXTERNAL_MASK, MASK_PATH
    # --- 1) INPUT_PATH ---
    if input_path is None:
        raise ValueError("resolve_runtime: 'input_path' is required.")
    INPUT_PATH = Path(input_path)

    # --- 2) VIEW ---
    if view is None or view not in ("side", "top"):
        VIEW = _detect_view_from_path(INPUT_PATH)
        if VIEW == "unknown":
            raise ValueError(
                "Cannot detect view type from input path. "
                "Please specify --view explicitly as 'side' or 'top'."
            )
    else:
        VIEW = view  

    # --- 3) OUTPUT_DIR ---
    OUTPUT_DIR = Path(output_dir) if output_dir else _default_output_dir(INPUT_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 4) External mask mode (ถ้าตั้ง MASK_PATH) ---
    USE_EXTERNAL_MASK = False
    if MASK_PATH:
        if not isinstance(MASK_PATH, Path):
            MASK_PATH = Path(MASK_PATH)
        if not MASK_PATH.exists():
            raise FileNotFoundError(f"External MASK_PATH not found: {MASK_PATH}")
        USE_EXTERNAL_MASK = True
        MASK_SPEC = None
        return

    # --- 5) ตั้ง MASK_SPEC ตาม VIEW (เมื่อไม่ได้ใช้ external mask) ---
    if VIEW == "side":
        '''MASK_SPEC = {
            "channel": "lab_a",
            "method": "binary",
            "threshold": 123,
            "object_type": "dark",
        }
        
        MASK_SPEC = {
            "channel": "lab_a",
            "method": "side_auto",  # ใช้ auto + guard
            "object_type": "dark",
            "s_min": 30,
            "v_bg": 45,
            "v_shadow_max": 115,
            "green_h": (10, 45),
            "bottom_roi_ratio": 0.60,
            "min_cc_area": 200,
        }
        
        '''
        MASK_SPEC = {
            "channel": "lab_a",
            "method": "side_auto",  # ใช้ auto + guard
            "object_type": "dark",
            "thresh_method": "fixed",
            "thresh_val": 120,
            
            "v_bg": 51,
            "s_min": 39,
            "v_shadow_max": 153,
            "green_h": (21, 90),
            
            "s_min_green": 10,
            "v_min_green": 45,
            
            "bottom_roi_ratio": 1.00,
            "min_cc_area": 200,
            "blur_ksize": 3,
            "open_ksize": 3,
            "close_ksize": 8,
        }
                
    elif VIEW == "top":
        '''
        MASK_SPEC = {
            "channel": "lab_a",
            "method": "otsu",
            "object_type": "dark",
            "ksize": 2001,  # ใช้เมื่อ method="gaussian"
            "offset": 5,    # ใช้เมื่อ method="gaussian"
        }
        '''
        MASK_SPEC = {
            "channel": "lab_a",
            "method": "side_auto",  # ใช้ auto + guard
            "object_type": "dark",
            "thresh_method": "fixed",
            "s_min": 30,
            "v_bg": 45,
            "v_shadow_max": 115,
            "green_h": (35, 95),
            "bottom_roi_ratio": 0.60,
            "min_cc_area": 200,
            "open_ksize": 3,
            "s_min_green": 10,
            "v_min_green": 45,
        }
        '''
        MASK_SPEC = {
            "channel": "lab_a",
            "method": "binary",
            "threshold": 120,
            "object_type": "dark",
        }
        '''
    else:
        MASK_SPEC = None  

# TOP view parameters
TOP_ROI_MODE = "manual"               # grid |auto | manual
TOP_SUMMARY_MODE = "per_object"     # per_object | per roi
TOP_MIN_PLANT_AREA = 4000      
TOP_MERGE_GAP = 0             # px; มาก = รวมง่าย
TOP_CLOSE_ITERS = 0            # ปิดรูเล็ก ๆ ก่อนจับกล่อง
TOP_ROI_X, TOP_ROI_Y, TOP_ROI_W, TOP_ROI_H = 1500, 650, 1000, 1000

STEM_CONNECT_MODE = "cc_touch"
STEM_CC_CLOSE_K   = 3

ROWS, COLS = 2, 4
ROI_TYPE = "partial"  # 'partial' | 'cutto' | 'largest'
ROI_RADIUS = 100
TOP_EXPECT_N_MIN = 4
TOP_EXPECT_N_MAX = 10

# ---------- STEM RESCUE (TOP view) ----------
#add v to connect top-view
ENABLE_STEM_RESCUE        = True   # เปิด/ปิดฟีเจอร์กู้ก้าน
STEM_V_METHOD = "fixed"            # "fixed"|"otsu"|"percentile"
STEM_V_MIN = 155                    # มืด 50 / สว่าง 100
STEM_V_MAX = 255
RESCUE_B_OTSU_FALLBACK    = 135
RESCUE_L_MIN_FOR_STEM     = 120
RESCUE_S_MIN_FOR_STEM     = 12
RESCUE_A_WHITE_MAX        = 150
RESCUE_HUE_YELLOW         = (15, 45)   # OpenCV H: 0–179
RESCUE_GEO_ITERS          = 40
RESCUE_VERT_DILATE_K      = 9
RESCUE_MIN_AREA_PX        = 800        # ปรับตามความละเอียดภาพ

# -------------------------------------------------
# SIDE view parameters (rectangle ROI)
SIDE_CROP_ENABLE = False         # เปิด/ปิดการครอปรูปก่อนหา ROI
SIDE_CROP_RECT   = (500, 250, 3000, 2000)  # (x, y, w, h)
SIDE_ROI_MODE = "manual"      # mode : "auto" | "manual"
USE_FULL_IMAGE_ROI = False
ROI_X, ROI_Y, ROI_W, ROI_H = 1500, 1250, 600, 1000
SIDE_TOUCH_RECT = (ROI_X, ROI_Y, ROI_W, ROI_H)
MIN_PLANT_AREA = 1000
SIDE_MERGE_GAP = 10
SIDE_EXPECT_N_MIN = 3
SIDE_EXPECT_N_MAX = 20
SIDE_BRIDGE_GAP_X = 10
SIDE_BRIDGE_GAP_Y = 10

ENABLE_SIDE_STEM_RESCUE = True  #เปิด/ปิดการกู้ลำต้นสำหรับ Side-view
SIDE_V_METHOD = "fixed"            # "fixed"|"otsu"|"percentile"
SIDE_V_MIN = 157                    # มืด 50 / สว่าง 100
SIDE_V_MAX = 255


# -------------------------------------------------
# Calibration scale
CHECKER_SQUARE_MM = 12.0 # Size of one square in checkerboard (mm)
RECT_SIZE_MM = (48, 48) # (w, h) ของสี่เหลี่ยมอ้างอิงจริง (mm)
CHECKER_PATTERN = (3, 3)
FALLBACK_MM_PER_PX = 48.0 / 490.0  # mm/px; กรณีไม่พบ checkerboard   
FALLBACK_MM_PER_PX_SIDE = 0.211
# -------------------------------------------------
# Measurement parameters side view
DEFAULT_CALIBRATION_OBJECT_DISTANCE_MM = 450.0  # mm
DEFAULT_PLANT_DISTANCE_MM = 330.0  # mm

#Distance map based on filename
SIDE_DISTANCE_MAP = {
    "_A_": (450.0, 330.0), # (Ref: 45cm, Plant: 33cm)
    "_B_": (450.0, 330.0),
    "_C_": (450.0, 330.0),
    
    "_D_": (450.0, 120.0), # (Ref: 45cm, Plant: 12cm)
    "_E_": (450.0, 120.0),
    "_F_": (450.0, 120.0)
}

# -------------------------------------------------
# Mask scoring weights
W_COVERAGE   = 1.0
W_COMPONENTS = 1.2
W_SOLIDITY   = 0.5
W_BORDER     = 1.0
FORCE_OBJECT_WHITE = True
COVERAGE_TARGET = 0.05
MERGE_COMPONENTS_PER_SLOT = True
