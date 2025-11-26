import cv2
import numpy as np
from plantcv import plantcv as pcv

def make_grid_rois(rgb_img, rows, cols, roi_radius=None, margin=0.05, shape="circle"):
    """
    สร้างกริด ROI (rows x cols) แล้วคืนเป็น list ของคอนทัวร์ (ใช้กับ cv2.drawContours)
    - วางศูนย์กลางกลางเซลล์ (offset 0.5)
    - clamp รัศมีตามระยะจาก "ศูนย์กลางสุดขอบ" ถึงขอบภาพ
    """
    H, W = rgb_img.shape[:2]
    if rows <= 0 or cols <= 0:
        raise ValueError("rows/cols must be > 0")

    # กรอบใช้งานหลังหัก margin
    mx, my = int(W * margin), int(H * margin)
    x0, y0 = mx, my
    x1, y1 = W - mx - 1, H - my - 1

    # ขนาดเซลล์ (กว้าง/สูง) = แบ่งพื้นที่ใช้งานด้วยจำนวนคอลัมน์/แถว
    DX = (x1 - x0) / float(cols)   # << ต่างจากเดิม: หารด้วย cols (ไม่ใช่ cols-1)
    DY = (y1 - y0) / float(rows)  # << ต่างจากเดิม: หารด้วย rows

    cx_min = x0 + 0.5 * DX
    cx_max = x0 + (cols - 0.5) * DX
    cy_min = y0 + 0.5 * DY
    cy_max = y0 + (rows - 0.5) * DY

    # รัศมีตั้งต้น
    if roi_radius is None:
        eff_r = int(max(2, round(0.45 * min(DX, DY))))
    else:
        eff_r = int(max(2, round(float(roi_radius))))

    # clamp รัศมีตาม "ระยะห่างของศูนย์กลางสุดขอบ" ถึงขอบภาพ
    eff_r = int(min(
        eff_r,
        cx_min,                 # ระยะจากศูนย์กลางซ้ายสุดถึงขอบซ้าย
        (W - 1) - cx_max,       # ระยะจากศูนย์กลางขวาสุดถึงขอบขวา
        cy_min,                 # ระยะจากศูนย์กลางบนสุดถึงขอบบน
        (H - 1) - cy_max        # ระยะจากศูนย์กลางล่างสุดถึงขอบล่าง
    ))
    if eff_r <= 0:
        raise ValueError("ROI grid does not fit the image. Adjust rows/cols/margin/radius.")

    # สร้างคอนทัวร์ ROI จากศูนย์กลาง "กลางเซลล์"
    rois = []
    for r in range(rows):
        cy = y0 + (r + 0.5) * DY
        for c in range(cols):
            cx = x0 + (c + 0.5) * DX
            if shape == "rect":
                half = eff_r
                cnt = np.array(
                    [[cx - half, cy - half],
                     [cx + half, cy - half],
                     [cx + half, cy + half],
                     [cx - half, cy + half]], dtype=np.int32
                ).reshape(-1, 1, 2)
            else:
                pts = cv2.ellipse2Poly((int(round(cx)), int(round(cy))),
                                       (eff_r, eff_r), 0, 0, 360, max(1, 360 // 96))
                cnt = pts.reshape(-1, 1, 2).astype(np.int32)
            rois.append(cnt)

    return rois, eff_r

def make_top_rois_auto(
    rgb_img,
    mask_fill,
    cfg=None,
    min_area_px=2000,
    close_iters=None,
    debug_out_path=None,
    merge_gap_px=None
):
    """
    เวอร์ชัน 'ปิดการรวมถาวร'
    - ROI = component ตรงๆ (หนึ่งคอมโพเนนต์ต่อหนึ่ง ROI)
    - ไม่ทำการ merge กล่อง ไม่พิจารณา gap/overlap ใดๆ ทั้งสิ้น
    คืนค่า: rois = [{"idx": i, "bbox": (x,y,w,h), "comp_mask": mask_single_cid, "cids":[cid]} ...]
    """

    # ===== config =====
    if cfg is not None:
        if min_area_px is None:
            min_area_px = int(getattr(cfg, "TOP_MIN_PLANT_AREA", getattr(cfg, "MIN_PLANT_AREA", 800)))
        if close_iters is None:
            close_iters = int(getattr(cfg, "TOP_CLOSE_ITERS", 1))
    else:
        min_area_px = int(800 if min_area_px is None else min_area_px)
        close_iters  = int(1   if close_iters  is None else close_iters)

    H, W = rgb_img.shape[:2]
    m = (mask_fill > 0).astype(np.uint8) * 255

    # ===== morphology (เล็กน้อยพอ ไม่ให้คอมโพเนนต์ติดกัน) =====
    if close_iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=close_iters)

    # ===== connected components =====
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    # ===== เก็บเฉพาะคอมโพเนนต์ที่ผ่านเกณฑ์ =====
    comps = []  # [{cid, bbox, area}]
    for cid in range(1, num):
        x, y, w, h, area = stats[cid]
        if area >= min_area_px:
            comps.append({"cid": cid, "bbox": (int(x), int(y), int(w), int(h)), "area": int(area)})

    if not comps:
        return []

    # ===== ไม่ merge: ROI = component ตรงๆ =====
    rois = []
    dbg = rgb_img.copy()
    for i, c in enumerate(sorted(comps, key=lambda d: d["bbox"][0]), start=1):
        x, y, w, h = c["bbox"]
        cid = c["cid"]

        # mask เฉพาะคอมโพเนนต์นี้เท่านั้น
        comp_mask = np.zeros((H, W), dtype=np.uint8)
        comp_mask[labels == cid] = 255

        rois.append({
            "idx": i,
            "bbox": (x, y, w, h),
            "comp_mask": comp_mask,
            "cids": [cid],
        })

        # debug box
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(dbg, f"#{i}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    if debug_out_path:
        cv2.imwrite(debug_out_path, dbg)

    return rois

def create_roi_top_partial(
    rgb_img,
    mask_fill,
    ROI_X, ROI_Y, ROI_W, ROI_H,
    debug_path=None
):
    if mask_fill.ndim == 3:
        mask_fill = cv2.cvtColor(mask_fill, cv2.COLOR_BGR2GRAY)
    bin_mask = (mask_fill > 0).astype(np.uint8) * 255
    H, W = bin_mask.shape[:2]
    
    x = max(0, min(int(ROI_X), W - 1))
    y = max(0, min(int(ROI_Y), H - 1))
    w = max(1, min(int(ROI_W), W - x))
    h = max(1, min(int(ROI_H), H - y))
    roi_rect = (x, y, w, h)
    
    ret = pcv.roi.rectangle(img=rgb_img, x=x, y=y, w=w, h=h)
    roi_obj = ret[0] if isinstance(ret, tuple) else ret
    
    filtered = pcv.roi.filter(mask=bin_mask, roi=roi_obj, roi_type='partial')
    
    if filtered.dtype != np.uint8:
        filtered = filtered.astype(np.uint8)
    if filtered.max() not in (1, 255):
        _, filtered = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        if filtered.max() == 1:
            filtered = filtered * 255
            
    cnts = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    
    dbg = rgb_img.copy()
    
    if not contours:
        cv2.rectangle(dbg, (x, y), (x + w - 1, y + h - 1), (255, 0, 0), 2)
        if debug_path:
            cv2.imwrite(debug_path, dbg)
        return [], roi_rect
    
    roi_mask = np.zeros_like(bin_mask, dtype=np.uint8)
    cv2.rectangle(roi_mask, (x, y), (x + w - 1, y + h - 1), 255, thickness=-1)
    
    out_rois = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1:
            continue
        
        cnt_mask = np.zeros_like(bin_mask, dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        
        intersect_mask = cv2.bitwise_and(cnt_mask, roi_mask)
        intersect_area = cv2.countNonZero(intersect_mask)
        overlap_ratio = intersect_area / (area + 1e-12)
        
        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        out_rois.append({
            "bbox": (x_, y_, w_, h_),
            "comp_mask": cnt_mask,
        })

        cv2.rectangle(dbg, (x_, y_), (x_ + w_ - 1, y_ + h_ - 1), (0, 255, 0), 2)
        cv2.putText(dbg, f"A={int(area)} ov={overlap_ratio:.2f}",
                    (x_, max(0, y_ - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
        
    if debug_path:
        cv2.imwrite(debug_path, dbg)
    return out_rois, roi_rect
    

