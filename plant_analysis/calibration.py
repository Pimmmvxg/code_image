import cv2
import numpy as np
from pathlib import Path
from . import config as cfg
from plantcv import plantcv as pcv
import math
from statistics import median

def _order_box(pts4: np.ndarray) -> np.ndarray:
    # รับ 4 จุด (x,y) -> เรียงเป็น TL, TR, BR, BL
    pts = np.array(pts4, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def side_lengths(box4: np.ndarray) -> tuple[float, float]:
    tl, tr, br, bl = box4
    w1 = np.linalg.norm(tr - tl)
    w2 = np.linalg.norm(br - bl)
    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)
    width_px  = 0.5 * (w1 + w2)
    height_px = 0.5 * (h1 + h2)
    return width_px, height_px

def scale_from_contours(
    th: np.ndarray,
    image: np.ndarray,
    rect_size_mm=cfg.RECT_SIZE_MM, # (w, h) ของสี่เหลี่ยมอ้างอิงจริง (mm)
    crop_top_ratio=0.7,
    min_area=300000,
    eps_fraction=0.04, # การลดจุดมุม ได้มุมเยอะเพิ่มค่า มุมน้อยลดค่า
    rect_tol=0.3, # ยอมให้ aspect ratio เพี้ยนจากของจริงได้
    min_rectangularity=0.6, # ความเป็นสี่เหลี่ยมขั้นต่ำ
    previous_scale=None,
    fallback_scale=cfg.FALLBACK_MM_PER_PX,
    save_debug=True,
    debug_name="rectangle_scale"
):
    # หา contours
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0
    rect_w_mm, rect_h_mm = float(rect_size_mm[0]), float(rect_size_mm[1])
    target_aspect = max(rect_w_mm, rect_h_mm) / max(1e-9, min(rect_w_mm, rect_h_mm))

    for c in cnts:
        area = cv2.contourArea(c)
        if area < float(min_area):
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, float(eps_fraction) * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        rect = cv2.minAreaRect(c)       # (center,(w,h),angle)
        (rw, rh) = rect[1]
        if rw <= 1 or rh <= 1:
            continue

        rect_area = rw * rh
        rectangularity = float(area) / float(rect_area + 1e-6)
        if rectangularity < float(min_rectangularity):
            continue

        aspect = max(rw, rh) / max(1e-9, min(rw, rh))
        ratio_err = abs(aspect - target_aspect) / target_aspect
        if ratio_err > float(rect_tol):
            continue

        score = rectangularity * (1.0 - ratio_err) * math.sqrt(max(area, 1.0))
        if score > best_score:
            best_score = score
            box = cv2.boxPoints(rect)
            box = _order_box(box)
            best = {"box": box, "area": area, "aspect": aspect, "ratio_err": ratio_err}

    if best is None:
        if previous_scale is not None and previous_scale > 0:
            return float(previous_scale), False, "rect: using previous scale"
        return float(fallback_scale), False, "rect: fallback scale"

    # 2) คำนวณ mm/px
    width_px, height_px = side_lengths(best["box"])
    long_px, short_px = (width_px, height_px) if width_px >= height_px else (height_px, width_px)
    long_mm, short_mm = (max(rect_w_mm, rect_h_mm), min(rect_w_mm, rect_h_mm))
    s_long  = long_mm  / max(long_px,  1e-9)
    s_short = short_mm / max(short_px, 1e-9)
    s_area  = math.sqrt((rect_w_mm * rect_h_mm) / max(best["area"], 1e-9))
    scale = float(median([s_long, s_short, s_area]))

    # 3) debug
    if save_debug and (pcv.params.debug_outdir or getattr(cfg, "OUTPUT_DIR", None)):
        vis = image.copy()
        tl, tr, br, bl = best["box"].astype(int)
        cv2.polylines(vis, [np.array([tl, tr, br, bl])], True, (0,255,255), 2)
        txt = f"mm/px={scale:.6f}"
        cv2.rectangle(vis, (10,10), (10+len(txt)*9, 40), (0,0,0), -1)
        cv2.putText(vis, txt, (14,32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        base = getattr(cfg, "OUTPUT_DIR", None) or pcv.params.debug_outdir or "."
        Path(base, "processed").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(base, "processed", f"{debug_name}.png")), vis)

    return scale, True, "scale from rectangle"

def get_scale_from_rectangle(
    image,
    rect_size_mm=cfg.RECT_SIZE_MM, # (w, h) ของสี่เหลี่ยมอ้างอิงจริง (mm)
    crop_top_ratio=1.0,
    min_area=9050000,
    eps_fraction=0.04, # การลดจุดมุม ได้มุมเยอะเพิ่มค่า มุมน้อยลดค่า
    rect_tol=0.3, # ยอมให้ aspect ratio เพี้ยนจากของจริงได้
    min_rectangularity=0.7, # ความเป็นสี่เหลี่ยมขั้นต่ำ
    previous_scale=None,
    fallback_scale=cfg.FALLBACK_MM_PER_PX,
    save_debug=True,
    debug_name="rectangle_scale"
):
    img = image
    if image is None:
        return (float(previous_scale) if previous_scale else float(fallback_scale)), False, "no image"
    
    H, W = img.shape[:2]
    roi = img[:int(H * crop_top_ratio)].copy()
    
    view = getattr(cfg, "VIEW", "top").lower()
    
    if view == "top":
        V = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:,:,2]  # Value channel
        th = pcv.threshold.binary(gray_img=V, threshold=190, object_type='light')
        
        return scale_from_contours(
            th,
            image,
            rect_size_mm=rect_size_mm,
            crop_top_ratio=crop_top_ratio,
            min_area=min_area,
            eps_fraction=eps_fraction,
            rect_tol=rect_tol,
            min_rectangularity=min_rectangularity,
            previous_scale=previous_scale,
            fallback_scale=fallback_scale,
            save_debug=save_debug,
            debug_name=f"{debug_name}_top"
        )
    else:  # side view
        '''
        L = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)[:,:,0]
        L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(L)
        _, th = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, 1)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, 1)
        '''
        # แปลง BGR → HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # ดึง V-channel (ความสว่าง)
        V = hsv[:,:,2]

        # threshold แบบ fixed = 150
        _, th = cv2.threshold(V, 150, 255, cv2.THRESH_BINARY)

        # kernel วงรี 5x5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

        # Closing: ปิดรูเล็กๆ
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, 1)

        # Opening: ลบ noise จุดเล็กๆ
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, 1)  
        
        return scale_from_contours(
            th,
            image,
            rect_size_mm=rect_size_mm,
            crop_top_ratio=crop_top_ratio,
            min_area=min_area,
            eps_fraction=eps_fraction,
            rect_tol=rect_tol,
            min_rectangularity=min_rectangularity,
            previous_scale=previous_scale,
            fallback_scale=cfg.FALLBACK_MM_PER_PX_SIDE,
            save_debug=save_debug,
            debug_name=f"{debug_name}_side"
        )

def get_scale_from_checkerboard(
    image,
    square_size_mm=cfg.CHECKER_SQUARE_MM,
    pattern_size=cfg.CHECKER_PATTERN,
    previous_scale=None,
    fallback_scale=cfg.FALLBACK_MM_PER_PX_SIDE,
    refine=True,
    save_debug=True,
    debug_name="checkerboard_scale"
):
    """
    คืน mm_per_px จาก checkerboard + flag ว่าพบหรือไม่ + ข้อความอธิบาย
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if found and refine:
        # ปรับจุดให้คมขึ้น
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)

    if found:
        # คำนวณระยะต่อช่องแบบ robust: เฉลี่ยทั้งแนว x และแนว y
        w, h = pattern_size
        # แนวนอน: แต่ละแถว มี (w-1) ช่อง
        horiz = []
        for r in range(h):
            for c in range(w - 1):
                p1 = corners[r * w + c][0]
                p2 = corners[r * w + (c + 1)][0]
                horiz.append(np.linalg.norm(p1 - p2))
        # แนวตั้ง: แต่ละคอลัมน์ มี (h-1) ช่อง
        vert = []
        for c in range(w):
            for r in range(h - 1):
                p1 = corners[r * w + c][0]
                p2 = corners[(r + 1) * w + c][0]
                vert.append(np.linalg.norm(p1 - p2))

        dpx = float(np.median(horiz + vert)) if (horiz or vert) else None
        if dpx and dpx > 0:
            scale = float(square_size_mm) / dpx   # mm/px
            scale_info = "scale from checkerboard"
            # debug image (optional)
            if save_debug and (pcv.params.debug_outdir or getattr(cfg, "OUTPUT_DIR", None)):
                dbg = image.copy()
                cv2.drawChessboardCorners(dbg, pattern_size, corners, True)
                base = getattr(cfg, "OUTPUT_DIR", None) or pcv.params.debug_outdir or "."
                Path(base, "processed").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(base, "processed", f"{debug_name}.png")), dbg)
            return scale, True, scale_info

    # ไม่เจอ checkerboard → ใช้ previous หรือ fallback
    if previous_scale is not None and previous_scale > 0:
        return float(previous_scale), False, "using previous scale"
    return float(fallback_scale), False, "fallback scale"
def get_scale(
    image,
    prefer=("rectangle", "checkerboard"),
    previous_scale=None,
    fallback_scale=cfg.FALLBACK_MM_PER_PX_SIDE,
    rectangle_kwargs=None,
    checker_kwargs=None,
):
    """
    ตัวรวมวิธีคาลิเบรตสเกลสำหรับ pipeline:
      - ลองตามลำดับ prefer; ถ้าพบอย่างใดอย่างหนึ่งให้คืนค่านั้นเลย
      - ไม่พบทั้งหมด -> previous หรือ fallback
    rectangle_kwargs: dict ของพารามิเตอร์ที่จะส่งเข้า get_scale_from_rectangle(...)
    checker_kwargs:   dict ของพารามิเตอร์ที่จะส่งเข้า get_scale_from_checkerboard(...)
    """
    rectangle_kwargs = rectangle_kwargs or {}
    checker_kwargs = checker_kwargs or {}

    for m in prefer:
        if m == "rectangle":
            s, ok, info = get_scale_from_rectangle(
                image,
                previous_scale=previous_scale,
                fallback_scale=fallback_scale,
                **rectangle_kwargs
            )
            if ok: return s, True, info
        elif m == "checkerboard":
            s, ok, info = get_scale_from_checkerboard(
                image,
                previous_scale=previous_scale,
                fallback_scale=fallback_scale,
                **checker_kwargs
            )
            if ok: return s, True, info

    # ไม่มีวิธีไหนเจอ
    if previous_scale is not None and previous_scale > 0:
        return float(previous_scale), False, "using previous scale"
    return float(fallback_scale), False, "fallback scale"