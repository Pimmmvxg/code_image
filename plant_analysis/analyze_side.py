from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path
from . import config as cfg

import os
import cv2
import numpy as np
from plantcv import plantcv as pcv

from . import config as cfg
from .masking import ensure_binary
from .color import get_color_name
from .movefile import enqueue_copy

# ---------------------------- Data class (optional) ---------------------------- #
@dataclass
class SideBBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min + 1)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min + 1)

# ---------------------------- Helpers ---------------------------- #
def _bbox_from_mask(m: np.ndarray) -> SideBBox:
    ys, xs = np.where(m > 0)
    if ys.size == 0 or xs.size == 0:
        return SideBBox(0, 0, 0, 0)
    return SideBBox(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

def _skeletonize(mask: np.ndarray) -> np.ndarray:
    return pcv.morphology.skeletonize(mask=mask)

def _prune_robust(skel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Iteratively prune; tolerate different PlantCV return signatures."""
    sizes = [10, 25, 50, 75, 100, 125, 150, 175, 200]
    pruned = skel.copy()
    for sz in sizes:
        try:
            ret = pcv.morphology.prune(skel_img=pruned, size=int(sz), mask=mask)
            if isinstance(ret, tuple):
                # (skel, seg_img, edge_objects) or (skel, edge_objects)
                pruned = ret[0]
            else:
                pruned = ret
        except Exception:
            pass
    return pruned

def _find_endpoints(skel: np.ndarray) -> np.ndarray:
    """Return Nx2 array of (y,x) coords for pixels with degree==1."""
    k = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D((skel > 0).astype(np.uint8), -1, k)
    ys, xs = np.where(nb == 11)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=int)
    return np.stack([ys, xs], axis=1)

def _px_to_mm(px: float) -> Optional[float]:
    """Convert pixel length to millimeters using cfg.MM_PER_PX (mm per pixel).
    If not available, return None (caller can skip writing mm values).
    """
    mm_per_px = getattr(cfg, "MM_PER_PX", None)
    if mm_per_px is None:
        return None
    try:
        mm_per_px = float(mm_per_px)
    except Exception:
        return None
    return float(px) * mm_per_px

def _save_debug(img: np.ndarray, name: str) -> None:
    try:
        base = Path(pcv.params.debug_outdir or ".")
        base.mkdir(parents=True, exist_ok=True)
        pcv.print_image(img=img, filename=str(base / name))
    except Exception:
        pass

def _draw_size_overlay(mask: np.ndarray, rgb: np.ndarray, bbox: SideBBox,
                       height_px: float, length_px: float,
                       height_mm: Optional[float], length_mm: Optional[float]) -> np.ndarray:
    vis = rgb.copy()
    # mask outline
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (250, 20, 20), 2)
    # bbox + dimensions text
    cv2.rectangle(vis, (bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max), (0, 255, 0), 2)
    x_mid = int(0.5*(bbox.x_min + bbox.x_max))
    cv2.line(vis, (x_mid, bbox.y_min), (x_mid, bbox.y_max), (0, 255, 0), 2)
    # Annotations
    txt1 = f"H: {height_px:.1f}px" + (f" ({height_mm:.1f} mm)" if height_mm is not None else "")
    txt2 = f"L: {length_px:.1f}px" + (f" ({length_mm:.1f} mm)" if length_mm is not None else "")
    # กำหนดจุดเริ่มต้น (บรรทัดแรก) ใต้ bbox
    y0 = min(vis.shape[0] - 20, bbox.y_max + 25)
    line_space = 50  # ระยะห่างระหว่างบรรทัด (px)

    # บรรทัดแรก
    cv2.putText(vis, txt1,
                (bbox.x_min, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (68, 0, 255), 2, cv2.LINE_AA)
    # บรรทัดสอง
    cv2.putText(vis, txt2,
                (bbox.x_min, min(vis.shape[0] - 5, y0 + line_space)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (68, 0, 255), 2, cv2.LINE_AA)
    
    return vis

def _draw_shape_overlay(rgb: np.ndarray,
                        shape_height_px: float, shape_length_px: float,
                        shape_height_mm: Optional[float], shape_length_mm: Optional[float]) -> np.ndarray:
    """Overlay showing image-shape-based size (H,W) regardless of mask.
    Draws a border around the full image and annotates H/W in px (+mm if available).
    """
    vis = rgb.copy()
    H, W = vis.shape[:2]
    # full-frame border
    cv2.rectangle(vis, (1,1), (W-2, H-2), (255, 0, 200), 2)
    # vertical
    cv2.arrowedLine(vis, (W-10, 5), (W-10, H-5), (255, 0, 200), 2, tipLength=0.02)
    # horizontal
    cv2.arrowedLine(vis, (5, H-10), (W-5, H-10), (255, 0, 200), 2, tipLength=0.02)
    # labels
    txtH = f"H_shape: {shape_height_px:.1f}px" + (f" ({shape_height_mm:.1f} mm)" if shape_height_mm is not None else "")
    txtW = f"W_shape: {shape_length_px:.1f}px" + (f" ({shape_length_mm:.1f} mm)" if shape_length_mm is not None else "")
    cv2.putText(vis, txtH, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,200), 2, cv2.LINE_AA)
    cv2.putText(vis, txtW, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,200), 2, cv2.LINE_AA)
    return vis

def analyze_one_side(slot_mask: np.ndarray, sample_name: str, rgb_img: np.ndarray) -> None:

    # Normalize mask to binary 0/255 with object=white
    slot_mask = ensure_binary(slot_mask, normalize_orientation=True)
    if slot_mask is None or cv2.countNonZero(slot_mask) == 0:
        raise ValueError("analyze_one_side: empty mask")

    H, W = slot_mask.shape[:2]

    # Save basic mask debug
    mask_vis = cv2.cvtColor(slot_mask, cv2.COLOR_GRAY2BGR)
    _save_debug(mask_vis, f"{sample_name}_side_mask.png")

    #Analyze Skeletonize & prune
    if getattr(cfg, "ENABLE_SKELETON_SIDE", False):
    
        # Skeletonize & prune
        skel = _skeletonize(slot_mask)
        pruned = _prune_robust(skel, slot_mask)

        # Save skeleton debug (raw + pruned)
        skel_vis = rgb_img.copy()
        yy, xx = np.where(skel > 0)
        skel_vis[yy, xx] = (0, 255, 255)
        _save_debug(skel_vis, f"{sample_name}_side_skeleton.png")

        pruned_vis = rgb_img.copy()
        yy, xx = np.where(pruned > 0)
        pruned_vis[yy, xx] = (255, 255, 0)
        _save_debug(pruned_vis, f"{sample_name}_side_skeleton_pruned.png")

        # Endpoints
        endpoints = _find_endpoints(pruned)
        n_endpoints = int(endpoints.shape[0])
        
        endpoints_vis = pruned_vis.copy()
        if n_endpoints >= 1:
            for (ey, ex) in endpoints:
                cv2.circle(endpoints_vis, (int(ex), int(ey)), 3, (0, 0, 255), -1)
        else:
            pcv.outputs.add_observation(sample=sample_name, variable="has_endpoints",
                                        trait="flag", method="skeleton_endpoints",
                                        scale="None", datatype=bool, value=False, label="has_endpoints")
        _save_debug(endpoints_vis, f"{sample_name}_side_endpoints.png")
    

    # Size from mask bbox
    bbox = _bbox_from_mask(slot_mask)
    height_px = float(bbox.height)
    length_px = float(bbox.width)

    # mm conversion via cfg.MM_PER_PX (mm/px)
    height_mm = _px_to_mm(height_px)
    length_mm = _px_to_mm(length_px)

    # Area in px
    area_px = int(cv2.countNonZero(slot_mask))

    # Extra: Shape-based size (image dimensions)
    shape_H, shape_W = slot_mask.shape[:2]
    shape_height_px = float(shape_H)
    shape_length_px = float(shape_W)
    shape_height_mm = _px_to_mm(shape_height_px)
    shape_length_mm = _px_to_mm(shape_length_px)
    
    # OVERLAY: size/shape summary (like analyze.size style)
    size_overlay = _draw_size_overlay(slot_mask, rgb_img, bbox, height_px, length_px, height_mm, length_mm)
    _save_debug(size_overlay, f"{sample_name}_side_size_overlay.png")
    # (A) OVERLAY: custom size/shape summary (like analyze.size style)
    shape_overlay = _draw_shape_overlay(rgb_img, shape_height_px, shape_length_px, shape_height_mm, shape_length_mm)
    _save_debug(shape_overlay, f"{sample_name}_side_shape_overlay.png")

    # (B) PlantCV built-in: pcv.analyze.size
    try:
        result = pcv.create_labels(bin_img=slot_mask)
        if isinstance(result, tuple):
            labeled_mask, n_labels = result
        else:
            labeled_mask = result
            n_labels = int(np.max(labeled_mask))
        pcv.analyze.size(img=rgb_img, labeled_mask=labeled_mask, label=sample_name)
    except Exception:
        pass
    
    # ---------------- Record observations ---------------- #
    def _add(var, value, trait, method, scale, dt, label=None):
        pcv.outputs.add_observation(sample=sample_name, variable=var, trait=trait,
                                    method=method, scale=scale, datatype=dt,
                                    value=value, label=label or var)

    _add("height_px", height_px, "height", "bbox_y_span", "px", float)
    _add("length_px", length_px, "length", "bbox_x_span", "px", float)
    _add("area_px", area_px, "projected_area", "mask_nonzero", "px^2", int)
    #_add("n_endpoints", n_endpoints, "skeleton_endpoints", "degree==1", "count", int)
    _add("bbox", f"({bbox.x_min},{bbox.y_min},{bbox.x_max},{bbox.y_max})", "bbox_xyxy", "mask_bbox", "px", str)

    if height_mm is not None:
        _add("height_mm", float(height_mm), "height", "px_to_mm", "mm", float)
    if length_mm is not None:
        _add("length_mm", float(length_mm), "length", "px_to_mm", "mm", float)
    
    # Record shape size too
    _add("shape_height_px", shape_height_px, "shape_height", "image_shape", "px", float)
    _add("shape_length_px", shape_length_px, "shape_length", "image_shape", "px", float)
    if shape_height_mm is not None:
        _add("shape_height_mm", shape_height_mm, "shape_height", "image_shape", "mm", float)
    if shape_length_mm is not None:
        _add("shape_length_mm", shape_length_mm, "shape_length", "image_shape", "mm", float)

    # Extra: save a consolidated debug with tiles
    try:
        gap = 8
        h2 = max(arr.shape[0] for arr in [rgb_img, size_overlay, endpoints_vis])
        w1 = rgb_img.shape[1]
        w2 = size_overlay.shape[1]
        w3 = endpoints_vis.shape[1]
        canvas = np.zeros((h2, w1 + w2 + w3 + gap*2, 3), dtype=np.uint8)
        canvas[:rgb_img.shape[0], :w1] = rgb_img
        canvas[:size_overlay.shape[0], w1+gap:w1+gap+w2] = size_overlay
        canvas[:endpoints_vis.shape[0], w1+gap+w2+gap:w1+gap+w2+gap+w3] = endpoints_vis
        _save_debug(canvas, f"{sample_name}_side_summary.png")
    except Exception:
        pass

def _color_name_under_mask(rgb_img: np.ndarray, mask: np.ndarray) -> tuple[str, float]:
    """คำนวณ median hue (degree 0..360) ใต้ mask แล้วแปลงเป็นชื่อสีเดียวกับ topview"""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    m = (mask > 0)
    if not np.any(m):
        return "Unknown", 0.0
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0][m].astype(np.float32) * 2.0 # [0..180] -> [0..360]
    hue_med = float(np.median(h)) if h.size > 0 else 0.0
    return get_color_name(hue_med), hue_med

def save_side_overlay(
    rgb_img: np.ndarray,
    slot_mask: np.ndarray,
    sample_name: str = "side",
    mm_per_px: Optional[float] = None,
) -> np.ndarray:
    """
    วาด overlay สำหรับ side view:
      - ใช้ size_overlay จาก _draw_size_overlay(...) เป็นพื้น
      - วาด contour (เหลือง), convex hull (ฟ้า), centroid (แดง)
      - กล่องข้อความสรุป: "Main Color: <name> | Area: <...>"
        * area แสดง mm*mm และ cm*cm ถ้ามี mm_per_px 
    """
    slot_mask = ensure_binary(slot_mask, normalize_orientation=True)
    if slot_mask is None or cv2.countNonZero(slot_mask) == 0:
        return rgb_img

    # 1) คำนวณขนาดจาก bbox
    bbox = _bbox_from_mask(slot_mask)
    height_px = float(bbox.height)
    length_px = float(bbox.width)
    # ใช้สเกลจากพารามิเตอร์ ถ้าไม่ได้ส่งมา จะ fallback ไปใช้ cfg.MM_PER_PX ภายใน _px_to_mm
    if mm_per_px is None:
        mm_per_px = getattr(cfg, "MM_PER_PX", None)
    if mm_per_px is not None:
        mm_per_px = float(mm_per_px)
        height_mm = height_px * mm_per_px
        length_mm = length_px * mm_per_px
    else:
        height_mm = None
        length_mm = None

    # 2) base = size overlay (เส้นกรอบ +ตัวเลข H,L)
    base = _draw_size_overlay(
        mask=slot_mask, rgb=rgb_img, bbox=bbox,
        height_px=height_px, length_px=length_px,
        height_mm=height_mm, length_mm=length_mm
    )
    overlay = base.copy()

    # 3) วาด contour + hull + centroid
    cnts, _ = cv2.findContours(slot_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        # ขอบวัตถุ
        cv2.drawContours(overlay, cnts, -1, (0, 255, 255), 2)
        # hull รวม
        all_pts = np.vstack(cnts)
        hull = cv2.convexHull(all_pts)
        if hull is not None and len(hull) >= 3:
            cv2.polylines(overlay, [hull], True, (255, 0, 255), 2)
    # centroid
    M = cv2.moments(slot_mask, binaryImage=True)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(overlay, (cx, cy), 4, (0, 0, 255), -1)
        
    #สี
    color_name, hue_med = _color_name_under_mask(rgb_img, slot_mask)

    # 4) กล่องข้อความสรุป (พื้นที่ + H×L)
    area_px = int(cv2.countNonZero(slot_mask))
    if mm_per_px is not None and mm_per_px > 0:
        area_mm2 = float(area_px) * (mm_per_px ** 2)
        area_cm2 = area_mm2 / 100.0
        area_text = f"{area_mm2:,.2f} mm² ({area_cm2:,.2f} cm²)"
    else:
        area_text = f"{area_px:,} px²"

    text = f"Main Color: {color_name} | Area: {area_text}"
    y0 = 30
    pad_w = max(10 + len(text) * 9, 260)
    cv2.rectangle(overlay, (10, y0 - 22), (10 + pad_w, y0 + 8), (0, 0, 0), -1)
    cv2.putText(overlay, text, (12, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    _save_debug(overlay, f"{sample_name}_side_overlay.png")
    return overlay

def get_side_legend(rgb_img: np.ndarray, mask: np.ndarray, label: str,
                    mm_per_px: Optional[float]) -> str:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    m = (mask > 0)
    area_px = int(np.count_nonzero(m))
    if mm_per_px and mm_per_px > 0:
        area_mm2 = float(area_px) * (mm_per_px ** 2)
        area_cm2 = area_mm2 / 100.0
        area_txt = f"{area_mm2:,.2f} mm*mm ({area_cm2:,.2f} cm*cm)"
    else:
        area_txt = f"{area_px:,} px*px"

    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0][m].astype(np.float32) * 2.0
    hue_med = float(np.median(h)) if h.size > 0 else 0.0
    color_name = get_color_name(hue_med)
    
    #H/L จาก bounding box ของก้อน
    bbox = _bbox_from_mask(mask)
    Hpx = float(bbox.height)
    Lpx = float(bbox.width)
    if mm_per_px:
        Htxt = f"{Hpx:.1f}px ({Hpx * mm_per_px:.1f} mm)"
        Ltxt = f"{Lpx:.1f}px ({Lpx * mm_per_px:.1f} mm)"
    else:
        Htxt = f"{Hpx:.1f}px"
        Ltxt = f"{Lpx:.1f}px"
    return f"{label}: Main Color: {color_name} | Area: {area_txt} | H: {Htxt} | L: {Ltxt}"

def combine_side_overlays(
    rgb_img: np.ndarray,
    masks: list[np.ndarray],
    labels: list[str],
    mm_per_px: Optional[float],
    out_path: Optional[str] = None
) -> np.ndarray:
    
    # รวม overlay ทุกต้นรวมเป็นภาพเดียว
    overlay = rgb_img.copy()

    for mask, label in zip(masks, labels):
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = ensure_binary(mask)
        if cv2.countNonZero(mask) == 0:
            continue
        # คำนวณ H/L ต่อก้อน + วาดลงภาพรวม
        bbox = _bbox_from_mask(mask)
        height_px = float(bbox.height)
        length_px = float(bbox.width)

        height_mm = length_mm = None
        if mm_per_px is not None and mm_per_px > 0:
            height_mm = height_px * float(mm_per_px)
            length_mm = length_px * float(mm_per_px)

        # วาดกรอบ + เส้นวัด + ตัวเลข H/L ลงบน "overlay" เฟรมรวม
        overlay = _draw_size_overlay(
            mask=mask, rgb=overlay, bbox=bbox,
            height_px=height_px, length_px=length_px,
            height_mm=height_mm, length_mm=length_mm
        )

        # กรอบ/ขอบวัตถุ
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(overlay, cnts, -1, (0, 255, 255), 2)  # เหลือง
            all_pts = np.vstack(cnts)
            hull = cv2.convexHull(all_pts)
            if hull is not None and len(hull) >= 3:
                cv2.polylines(overlay, [hull], True, (255, 0, 255), 2)  # ม่วง
        # centroid
        M = cv2.moments(mask, binaryImage=True)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(overlay, (cx, cy), 4, (0, 0, 255), -1)

    # แถบสรุปด้านบนรวมทุกก้อน
    legend_lines = []
    for mask, label in zip(masks, labels):
        legend_lines.append(get_side_legend(rgb_img, mask, label, mm_per_px))

    if legend_lines:
        pad_x, pad_y = 12, 10
        line_h = 38
        bar_h = pad_y * 2 + line_h * len(legend_lines)
        bar_w = max(overlay.shape[1], int(12 + max(len(s) for s in legend_lines) * 9))
        bar = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
        for i, line in enumerate(legend_lines):
            y = pad_y + (i + 1) * line_h - 10
            cv2.putText(bar, line, (pad_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        out = np.zeros((bar_h + overlay.shape[0], max(bar_w, overlay.shape[1]), 3), dtype=np.uint8)
        out[:bar_h, :bar_w] = bar
        out[bar_h:bar_h + overlay.shape[0], :overlay.shape[1]] = overlay
    else:
        out = overlay

    _target_name = cfg.safe_target_name(cfg.INPUT_PATH)
    if out_path is None:
        out_dir = pcv.params.debug_outdir or "./processed"
        os.makedirs(out_dir, exist_ok=True)
        out_path = str(Path(out_dir) / f"results_{_target_name}.png")
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        
    extra_dir = r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_result_sideview_smartfarm"
    os.makedirs(extra_dir, exist_ok=True)
    
    extra_path = str(Path(extra_dir) / f"results_{_target_name}.png")

    try:
        ok1 = cv2.imwrite(out_path, out)
        if not ok1:
            pcv.print_image(img=out, filename=out_path)
    except Exception as e:
        print(f"Warning: Cannot save local side overlay. out_path={out_path!r}, error={e}")
        return out

    # คิวก็อปขึ้น R: แบบ background
    try:
        enqueue_copy(out_path, extra_path, retries=3, backoff=1.6)
    except Exception as e:
        print(f"[MOVE TO R:] enqueue failed: src={out_path!r}, dst={extra_path!r}, err={e}")

    return out