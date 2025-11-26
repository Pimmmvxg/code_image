import numpy as np
import cv2
from plantcv import plantcv as pcv
from .color import get_color_name
from .movefile import enqueue_copy
from typing import Optional, List, Tuple
from . import config as cfg
from pathlib import Path

import json
import os
import re
import glob
import math

def _safe_starts(arr):
    if arr is None or len(arr) == 0:
        return (0.0, 0.0, 0.0)
    return (float(np.mean(arr)), float(np.std(arr)), float(np.median(arr)))

def add_global_density_and_color(rgb_img, mask_fill):
    total_px = mask_fill.size
    white_px = int(cv2.countNonZero(mask_fill))
    coverage_ratio = white_px / max(total_px, 1)
    _fc = cv2.findContours(mask_fill.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = _fc[2] if len(_fc) == 3 else _fc[0]
    n_comp = len(contours)
    areas = [cv2.contourArea(c) for c in contours] if contours else []
    a_med = float(np.median(areas)) if areas else 0.0
    a_mean = float(np.mean(areas)) if areas else 0.0

    if areas:
        big = contours[int(np.argmax(areas))]
        hull = cv2.convexHull(big)
        a_obj  = float(cv2.contourArea(big))
        a_hull = float(cv2.contourArea(hull)) if hull is not None else 0.0
        big_solidity = (a_obj / a_hull) if a_hull > 0 else 0.0
    else:
        big_solidity = 0.0
        
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    h, s, v = [ch[mask_fill > 0] for ch in cv2.split(hsv)]
    h_mean, h_std, h_med = _safe_starts(h.astype(np.float32) * 2.0)
    s_mean, s_std, s_med = _safe_starts(s.astype(np.float32))
    v_mean, v_std, v_med = _safe_starts(v.astype(np.float32))
    main_color = get_color_name(h_med)
    
    pcv.outputs.add_observation(sample='default', variable='all_coverage_ratio',
                                trait='ratio', method='count_nonzero/size', scale='none',
                                datatype=float, value=float(coverage_ratio), label='coverage')
    pcv.outputs.add_observation(sample='default', variable='all_n_components',
                                trait='count', method='findContours', scale='count',
                                datatype=int, value=int(n_comp), label='components')
    pcv.outputs.add_observation(sample='default', variable='all_comp_area_mean',
                                trait='area', method='mean(contourArea)', scale='px',
                                datatype=float, value=float(a_mean), label='mean component area')
    pcv.outputs.add_observation(sample='default', variable='all_comp_area_median',
                                trait='area', method='median(contourArea)', scale='px',
                                datatype=float, value=float(a_med), label='median component area')
    pcv.outputs.add_observation(sample='default', variable='all_big_solidity',
                                trait='ratio', method='largest(area)/convexHull', scale='none',
                                datatype=float, value=float(big_solidity), label='largest solidity')
    
    for (nm, val) in [
        ('hue_mean', h_mean), ('hue_std', h_std), ('hue_median', h_med),
        ('saturation_mean', s_mean), ('saturation_std', s_std), ('saturation_median', s_med),
        ('value_mean', v_mean), ('value_std', v_std), ('value_median', v_med)
    ]:
        pcv.outputs.add_observation(sample='default', variable=f'global_{nm}',
                                    trait='color', method='HSV stats(masked)', scale='unit',
                                    datatype=float, value=float(val), label=nm)
    pcv.outputs.add_observation(sample='default', variable='global_color_name',
                                trait='text', method='HSV median(name)', scale='none',
                                datatype=str, value=main_color, label='color name')

def analyze_one_top(slot_mask, sample_name, eff_r, rgb_img): 
    if slot_mask is None:
        raise RuntimeError("slot_mask is None (top)")
    if slot_mask.ndim == 3:
        slot_mask = cv2.cvtColor(slot_mask, cv2.COLOR_BGR2GRAY)
    if slot_mask.dtype != np.uint8:
        slot_mask = slot_mask.astype(np.uint8)
    slot_mask = np.where(slot_mask > 0, 255, 0).astype(np.uint8)
    
    area_px = int(cv2.countNonZero(slot_mask))
    roi_area_est = float(np.pi * (eff_r ** 2)) if (eff_r is not None and eff_r > 0) else 0.0
    coverage_local = (area_px / roi_area_est) if roi_area_est > 0 else 0.0
    
    _fc = cv2.findContours(slot_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = _fc[2] if len(_fc) == 3 else _fc[0]

    perim = 0.0
    convex_ratio = 0.0
    circularity = 0.0
    extent = 0.0
    n_comp = 0

    if contours:
        n_comp = len(contours)
        area_union = float(cv2.countNonZero(slot_mask))  # union area

        # เส้นรอบรูป = ผลรวมของทุกคอนทัวร์
        perim_sum = float(sum(cv2.arcLength(c, True) for c in contours))
        perim = perim_sum

        # hull รวม = hull(all_points) ;รูปหลายเหลี่ยมที่คลุมวัคถุ
        all_pts = np.vstack(contours)
        hull = cv2.convexHull(all_pts)
        a_hull = float(cv2.contourArea(hull)) if len(hull) >= 3 else 0.0
        convex_ratio = (area_union / a_hull) if a_hull > 0 else 0.0  # solidity ของ union

        # bbox รวมจาก hull
        x, y, w, h = cv2.boundingRect(hull)
        extent = (area_union / float(w * h)) if w * h > 0 else 0.0

        # circularity ของ union
        circularity = (4.0 * np.pi * area_union / (perim_sum ** 2)) if perim_sum > 0 else 0.0
        
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_idx = slot_mask > 0
    hh = h[mask_idx].astype(np.float32) * 2.0
    ss = s[mask_idx].astype(np.float32)
    vv = v[mask_idx].astype(np.float32)
    hue_med = float(np.median(hh)) if hh.size > 0 else 0.0
    col_name = get_color_name(hue_med)  
    
    pairs = [
        ('slot_area_px', area_px, 'area', 'px'),
        ('slot_coverage_ratio', coverage_local, 'ratio', 'none'),
        ('slot_perimeter', perim, 'length', 'px'),
        ('slot_convex_ratio', convex_ratio, 'ratio', 'none'),
        ('slot_circularity', circularity, 'ratio', 'none'),
        ('slot_extent', extent, 'ratio', 'none'),
        ('slot_n_components', float(n_comp), 'count', 'none'),
        ('hue_median', hue_med, 'color', 'unit'),
        ('sat_mean', float(np.mean(ss)) if ss.size>0 else 0.0, 'color', 'unit'),
        ('val_mean', float(np.mean(vv)) if vv.size>0 else 0.0, 'color', 'unit'),
    ]

    for var, val, trait, scale in pairs:
        pcv.outputs.add_observation(sample=sample_name, variable=var,
                                    trait=trait, method='no-skeleton stats', scale=scale,
                                    datatype=float, value=float(val), label=var)
    pcv.outputs.add_observation(sample=sample_name, variable='color_name',
                                trait='text', method='hue_median→name', scale='none',
                                datatype=str, value=col_name, label='dominant color')
    
def save_top_overlay(
    rgb_img,
    slot_mask,
    contours=None,
    eff_r=None,
    sample_name="default",
    mm_per_px: float | None = None,
    slot_label: str | None = None,
) -> str:

    if slot_mask is None:
        raise RuntimeError("save_top_overlay: slot_mask is None")
    # ให้เป็น binary uint8 0/255
    if slot_mask.ndim == 3:
        slot_mask = cv2.cvtColor(slot_mask, cv2.COLOR_BGR2GRAY)
    if slot_mask.dtype != np.uint8:
        slot_mask = slot_mask.astype(np.uint8)
    slot_mask = np.where(slot_mask > 0, 255, 0).astype(np.uint8)

    H, W = rgb_img.shape[:2]
    vis = rgb_img.copy()

    # หา contours ถ้าไม่ได้ส่งมา
    if contours is None:
        _fc = cv2.findContours(slot_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = _fc[2] if len(_fc) == 3 else _fc[0]

    # วาดคอนทัวร์ทุกก้อน (เขียว)
    if contours:
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    # รวมทุกจุดของทุกคอนทัวร์เพื่อทำ hull/bbox รวม
    all_pts = None
    if contours:
        all_pts = np.vstack(contours) if len(contours) > 0 else None

    hull = None
    if all_pts is not None and len(all_pts) >= 3:
        hull = cv2.convexHull(all_pts)
        # วาด hull (ฟ้า)
        cv2.drawContours(vis, [hull], -1, (255, 0, 0), 2)
        # วาด bbox ของ hull (เหลือง)
        x, y, w, h = cv2.boundingRect(hull)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # centroid ของ union mask
    M = cv2.moments(slot_mask, binaryImage=True)
    cx = cy = None
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)  # แดง
        if eff_r is not None and eff_r > 0:
            cv2.circle(vis, (cx, cy), int(eff_r), (255, 0, 255), 2)  # วงกลม ROI (ม่วง)

    # พื้นที่/สีหลักในมาสก์
    area_px = int(cv2.countNonZero(slot_mask))
    area_text = f"{area_px:,} px*px"
    area_mm2 = None
    if mm_per_px is not None and mm_per_px > 0:
        area_mm2 = float(area_px) * (mm_per_px ** 2)
        cm2 = area_mm2 / 100.0
        area_text = f"{area_mm2:,.2f} mm*mm ({cm2:,.2f} cm*cm)"

    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_idx = slot_mask > 0
    if np.count_nonzero(mask_idx) > 0:
        hue_deg = (h[mask_idx].astype(np.float32) * 2.0)
        hue_med = float(np.median(hue_deg))
        color_name = get_color_name(hue_med)
    else:
        color_name = "unknown"

    # ข้อความสรุปบนสุด
    label = slot_label or str(sample_name)
    text = f"{label} | Main Color: {color_name} | Area: {area_text}"
    y0 = 30
    pad_w = max(10 + len(text) * 9, 560)  # กันข้อความโดนตัด
    cv2.rectangle(vis, (10, y0 - 22), (10 + pad_w, y0 + 8), (0, 0, 0), -1)  # พื้นดำทึบ
    cv2.putText(vis, text, (12, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    filename = Path(sample_name).stem
    cv2.putText(vis, f"File: {filename}", (12, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    # เตรียมที่เซฟ
    out_dir = pcv.params.debug_outdir or "./processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sample_name}_top_overlay.png")

    # เซฟภาพ
    try:
        cv2.imwrite(out_path, vis)
    except Exception:
        pcv.print_image(img=vis, filename=out_path)

    # เซฟ meta (เผื่อใช้รวม/รีพอร์ต)
    meta = {
        "sample": sample_name,
        "slot_label": label,
        "area_px": int(area_px),
        "area_mm2": float(area_mm2) if area_mm2 is not None else None,
        "color_name": color_name,
        "eff_r": float(eff_r) if eff_r is not None else None,
        "overlay_path": out_path,
    }
    try:
        with open(os.path.join(out_dir, f"{sample_name}_top_overlay.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return out_path

def combine_top_overlays(
    rgb_img: np.ndarray,
    slot_masks: List[np.ndarray],
    labels: Optional[List[str]] = None,
    eff_r: Optional[float] = None,
    mm_per_px: Optional[float] = None,
    out_path: Optional[str] = None,
) -> Optional[str]:
    """
    รวม overlay ของทุกต้นให้อยู่ในภาพเดียว:
      - วาด contour ของแต่ละต้น (เขียว)
      - วาง label แต่ละต้นบนจุด centroid
      - สร้าง union mask เพื่อ:
          * หา convex hull รวม (ฟ้า)
          * วาด bbox รวม (เหลือง)
          * วาง centroid รวม (แดง)
      - คำนวณ 'Main Color + Area' ของแต่ละต้น แล้วสรุปเป็นแผงข้อความด้านบน:
          'slot name | Main Color | Area'
    """
    try:
        from .color import get_color_name
    except Exception:
        def get_color_name(_): return "unknown"

    if rgb_img is None or not slot_masks:
        return None

    # เตรียมภาพ/มาสก์
    overlay = rgb_img.copy()
    try:
        fname = Path(cfg.INPUT_PATH).stem
    except Exception:
        fname = "unknown"
    cv2.putText(overlay, f"File: {fname}", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    H, W = overlay.shape[:2]
    union_mask = np.zeros((H, W), dtype=np.uint8)
    all_contours_pts = []

    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    Hc, Sc, Vc = cv2.split(hsv)

    # เก็บบรรทัดสรุปต่อ 'ต้น'
    legend_lines = []

    # ช่วยแปลง/วัดพื้นที่
    def _ensure_bin(m: np.ndarray) -> np.ndarray:
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        if m.dtype != np.uint8:
            m = m.astype(np.uint8)
        return np.where(m > 0, 255, 0).astype(np.uint8)

    def _area_text(px_area: int) -> str:
        if mm_per_px is not None and mm_per_px > 0:
            mm2 = float(px_area) * (mm_per_px ** 2)
            cm2 = mm2 / 100.0
            return f"{mm2:,.2f} mm*mm ({cm2:,.2f} cm*cm)"
        return f"{px_area:,} px*px"

    # วาดต่อ-ต้น
    for i, m in enumerate(slot_masks):
        if m is None:
            continue
        m = _ensure_bin(m)
        if cv2.countNonZero(m) == 0:
            continue
        # รวม
        union_mask = cv2.bitwise_or(union_mask, m)

        # คอนทัวร์สีเขียว
        _fc = cv2.findContours(m.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = _fc[2] if len(_fc) == 3 else _fc[0]
        if contours:
            this_pts = np.vstack(contours)
            if this_pts is not None and len(this_pts) >= 3:
                this_hull = cv2.convexHull(this_pts)
                if this_hull is not None and len(this_hull) >= 3:
                    cv2.drawContours(overlay, [this_hull], -1, (60, 255, 0), 3)      # green
                    x, y, w, h = cv2.boundingRect(this_hull)
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 255), 3)  # เหลือง

        # centroid แต่ละต้น + label
        M = cv2.moments(m, binaryImage=True)
        cx = cy = None
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(overlay, (cx, cy), 3, (0, 255, 255), -1)  # จุด centroid (เหลือง)
            if eff_r is not None and eff_r > 0:
                cv2.circle(overlay, (cx, cy), int(eff_r), (255, 0, 255), 3)
                
        _ret = cv2.findContours(m.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = _ret[0] if len(_ret) == 2 else _ret[1]
        if contours:
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 3)

        # ค่าสี/พื้นที่ต่อ-ต้น
        mask_idx = m > 0
        area_px = int(cv2.countNonZero(m))
        if np.count_nonzero(mask_idx) > 0:
            hue_deg = (Hc[mask_idx].astype(np.float32) * 2.0)
            hue_med = float(np.median(hue_deg))
            color_name = get_color_name(hue_med)
        else:
            color_name = "unknown"

        slot_name = (labels[i] if labels and i < len(labels) and labels[i] else f"obj{i+1}")
        area_str = _area_text(area_px)

        # เขียน label ข้าง centroid ของต้นนั้น
        if cx is not None and cy is not None:
            small = f"{slot_name}"
            cv2.putText(overlay, small, (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 255), 4, cv2.LINE_AA)

        # เก็บเข้ารายการ legend
        legend_lines.append(f"{slot_name} | Main Color: {color_name} | Area: {area_str}")

    # สรุปด้านบน (1 ต้น = 1 บรรทัด)
    if legend_lines:
        pad_x = 12
        pad_y = 10
        line_h = 52
        bar_h = pad_y * 2 + line_h * len(legend_lines)
        bar_w = max(560, int(12 + max(len(s) for s in legend_lines) * 9))

        # ขยายแคนวาสด้านบนเพื่อใส่ข้อความสรุป
        new_canvas = np.zeros((bar_h + H, max(bar_w, W), 3), dtype=np.uint8)
        new_canvas[:bar_h, :bar_w] = (0, 0, 0)  # พื้นดำทึบ
        new_canvas[bar_h:bar_h + H, :W] = overlay
        overlay = new_canvas

        # เขียนข้อความสรุป
        for i, line in enumerate(legend_lines):
            y = pad_y + (i + 1) * line_h - 6
            cv2.putText(overlay, line, (pad_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
        
    extra_dir = r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_result_topview_smartfarm"
    os.makedirs(extra_dir, exist_ok=True)

    _target_name = cfg.safe_target_name(cfg.INPUT_PATH)
    extra_path = str(Path(extra_dir) / f"results_{_target_name}.png")

    if out_path is None:
        out_dir = pcv.params.debug_outdir or "./processed"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"results_{_target_name}.png")
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        ok1 = cv2.imwrite(out_path, overlay)
        if not ok1:
            pcv.print_image(img=overlay, filename=out_path)
    except Exception as e:
        print(f"Warning: Cannot write local overlay image. out_path={out_path!r}, error={e}")
        return out_path

    # คิวก็อปขึ้น R: แบบ background (ไม่บล็อก pipeline)
    try:
        enqueue_copy(out_path, extra_path, retries=3, backoff=1.6)
    except Exception as e:
        print(f"[MOVE TO R:] enqueue failed: src={out_path!r}, dst={extra_path!r}, err={e}")

    return out_path
