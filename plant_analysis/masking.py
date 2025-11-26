# masking.py
import cv2
import numpy as np
from plantcv import plantcv as pcv
from pathlib import Path
from . import config as cfg
from .custom_masking import auto_thresh_lab_a_otsu_guard

# ---------- Core helpers ----------
def _per_mm_from_shape(H, W):
    """คืนขนาดภาพเป็น mm ถ้ามี MM_PER_PX กำหนด"""
    mm_per_px = getattr(cfg, "MM_PER_PX", None)
    if mm_per_px:
        return float(H) * float(mm_per_px), float(W) * float(mm_per_px)
    return None, None

def ensure_binary(mask, normalize_orientation: bool = True):
    """
    ทำให้มาสก์เป็นภาพช่องเทาไบนารี 0/255
    และ (ค่าเริ่มต้น) เลือก orientation ที่เหมาะกว่าโดยอัตโนมัติ
    ให้ "วัตถุ = ขาว (255), พื้นหลัง = ดำ (0)"
    """
    if mask is None:
        return None
    m = np.asarray(mask)

    # บังคับเป็นช่องเทา uint8
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)

    # บีบให้มีแค่ 0/255
    m = np.where(m > 0, 255, 0).astype(np.uint8)

    if not normalize_orientation:
        return m

    # ลองกลับด้าน แล้วเลือกด้านที่ "สมเหตุสมผล" กว่า
    #m_inv = cv2.bitwise_not(m)
    m_inv = m

    def _score(mm: np.ndarray) -> float:
        H, W = mm.shape[:2]
        area_total = max(H * W, 1)
        ratio = float(cv2.countNonZero(mm)) / area_total

        # นับคอนทัวร์ + วัด solidity ของก้อนใหญ่สุด
        _fc = cv2.findContours(mm.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = _fc[2] if len(_fc) == 3 else _fc[0]
        n_comp = len(contours)
        if n_comp > 0:
            areas = [cv2.contourArea(c) for c in contours]
            cnt = contours[int(np.argmax(areas))]
            hull = cv2.convexHull(cnt)
            a_obj = float(cv2.contourArea(cnt))
            a_hull = float(cv2.contourArea(hull)) if hull is not None else 0.0
            solidity = (a_obj / a_hull) if a_hull > 0 else 0.0
        else:
            solidity = 0.0
            
         # ---------- VIEW-aware component preference ----------
        view = str(getattr(cfg, "VIEW", "top")).lower()
        if view == "top":
            exp_min = int(getattr(cfg, "TOP_EXPECT_N_MIN", 5))
            exp_max = int(getattr(cfg, "TOP_EXPECT_N_MAX", 6))
        else:  # side
            exp_min = int(getattr(cfg, "SIDE_EXPECT_N_MIN", 1))
            exp_max = int(getattr(cfg, "SIDE_EXPECT_N_MAX", 1))

        # ให้รางวัลเมื่อจำนวนก้อนอยู่ในช่วงที่คาดหวัง, ลงโทษเมื่อห่างออกไป
        if n_comp == 0:
            comp_score = -1.0
        elif n_comp < exp_min:
            comp_score = -0.2 * float(exp_min - n_comp)
        elif n_comp > exp_max:
            comp_score = -0.2 * float(n_comp - exp_max)
        else:
            comp_score = +0.3  # อยู่ในช่วงพอดี

        # เกณฑ์รวม: coverage ใกล้ 0.30 ดี + คะแนนตามจำนวนก้อน + solidity สูงดี
        #coverage_score = (1.0 - abs(0.30 - ratio))
        target = float(getattr(cfg, "MASK_TARGET_COVERAGE", 0.05))
        coverage_score = (1.0 - abs(target - ratio))

        top_touch    = np.count_nonzero(mm[0, :] > 0)
        bottom_touch = np.count_nonzero(mm[-1, :] > 0)
        left_touch   = np.count_nonzero(mm[:, 0] > 0)
        right_touch  = np.count_nonzero(mm[:, -1] > 0)

        # คิดเป็นสัดส่วนต่อความยาวขอบ
        top_p    = top_touch    / max(W, 1)
        bottom_p = bottom_touch / max(W, 1)
        left_p   = left_touch   / max(H, 1)
        right_p  = right_touch  / max(H, 1)

        # แตะทั้งหมดกี่ด้าน
        n_edges = sum(p > 0.5 for p in (top_p, bottom_p, left_p, right_p))
        # แตะหนักแค่ไหนรวมกัน (0..4)
        edge_strength = top_p + bottom_p + left_p + right_p

        # หักคะแนนตามจำนวนด้านที่แตะ + ความยาวที่แตะ
        border_penalty = 0.4 * n_edges + 0.6 * edge_strength
       
        return (cfg.W_COVERAGE * coverage_score
        + cfg.W_COMPONENTS * comp_score
        + cfg.W_SOLIDITY * solidity
        - cfg.W_BORDER * border_penalty)

    return m if _score(m) >= _score(m_inv) else m_inv

def _keep_top_k_components(m: np.ndarray, k: int) -> np.ndarray:
    """เก็บ k ก้อนใหญ่ที่สุด (connected components)"""
    m = ensure_binary(m, normalize_orientation=False)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    # label 0 = พื้นหลัง
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]   # ใหญ่ -> เล็ก
    keep = order[:k] + 1              # ชดเชย background
    out = np.zeros_like(m)
    for lab in keep:
        out[labels == lab] = 255
    return ensure_binary(out, normalize_orientation=False)

def connect_mask_holes(mask: np.ndarray,
                       gap_x: int = 100,
                       gap_y: int = 200,
                       iterations: int = 1) -> np.ndarray:
    
    m = ensure_binary(mask, normalize_orientation=False)
    if gap_x and gap_x > 0:
        kx = cv2.getStructuringElement(cv2.MORPH_RECT, (2*int(gap_x)+1, 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kx, iterations=iterations)
    if gap_y and gap_y > 0:
        ky = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2*int(gap_y)+1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ky, iterations=iterations)
    return ensure_binary(m, normalize_orientation=False)
'''
def prune_cross_bridge(m_before, m_after,
                       max_added_area=1500,
                       max_bridge_thickness=20):
    before = (m_before > 0).astype(np.uint8)
    after  = (m_after  > 0).astype(np.uint8)
    added  = ((after == 1) & (before == 0)).astype(np.uint8)

    # label ของมาสก์เดิม
    _, lbl_before = cv2.connectedComponents(before)

    # label ของส่วนที่เพิ่ม
    n_add, lbl_add = cv2.connectedComponents(added)
    keep = np.zeros_like(added, np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for i in range(1, n_add):
        blob = (lbl_add == i).astype(np.uint8)
        area = int(cv2.countNonZero(blob))
        if area == 0 or area > int(max_added_area):
            continue  # ก้อนที่เพิ่มใหญ่ผิดปกติ มักเป็น over-merge

        # ขยายขอบเล็กน้อยเพื่อเช็คว่าบล็อบไปแตะกี่คอมโพเนนต์เดิม
        touch = cv2.dilate(blob, k3, 1)
        touching_ids = np.unique(lbl_before[touch.astype(bool)])
        touching_ids = touching_ids[touching_ids != 0]

        # ไม่อนุญาตให้เป็นสะพานที่แตะ ≥2 คอมโพเนนต์
        if len(touching_ids) > 1:
            continue

        # กันตัวเชื่อมที่ "หนาเกินไป"
        ys, xs = np.where(blob)
        if xs.size:
            w = xs.max() - xs.min() + 1
            h = ys.max() - ys.min() + 1
            if min(w, h) > int(max_bridge_thickness):
                continue

        keep = cv2.bitwise_or(keep, blob)

    out = ((before == 1) | (keep == 1)).astype(np.uint8) * 255
    return out
    '''

def clean_mask(m, close_ksize=3, min_obj_size=500, keep_largest=False, keep_top_k=None):
    """
    ทำความสะอาดมาสก์:
    - ปิดรู/เชื่อมช่องว่างด้วย morphological close (ขนาด kernel = close_ksize)
    - กรองสิ่งเล็กกว่า min_obj_size ออก (pcv.fill)
    - (ทางเลือก) คงเฉพาะก้อนใหญ่สุด หรือ k ก้อนใหญ่สุด
    - บังคับ orientation สุดท้ายเป็น วัตถุ=ขาว
    """
    if m is None:
        return m

    # Normalize
    m = ensure_binary(m, normalize_orientation=True)

    # 1) close
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_ksize), int(close_ksize)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    # 2) กรองจุดเล็ก/อุดรู
    m = pcv.fill(bin_img=m, size=int(min_obj_size))

    # 3) เลือกเฉพาะก้อนใหญ่ (ถ้าต้องการ)
    if keep_top_k is not None and int(keep_top_k) >= 1:
        m = _keep_top_k_components(m, int(keep_top_k))
    elif keep_largest:
        m = _keep_top_k_components(m, 1)

    # ย้ำ orientation หลัง post-process
    return ensure_binary(m, normalize_orientation=True)

def auto_select_mask(rgb_img):
    """
    เลือกมาสก์อัตโนมัติจากหลายช่องสี/วิธี threshold
    คืนค่า: (mask_binary_normalized, meta_dict)
    """
    H, W = rgb_img.shape[:2]
    area_total = H * W
    height_shape = H
    width_shape = W

    # 1) เตรียมช่องสี (LAB/HSV) + rescale ปลอดภัย
    chans = []
    for nm, fn, kw in [
        ("lab_a", pcv.rgb2gray_lab, {'channel': 'a'}),
        ("lab_b", pcv.rgb2gray_lab, {'channel': 'b'}),
        ("lab_l", pcv.rgb2gray_lab, {'channel': 'l'}),
        ("hsv_v", pcv.rgb2gray_hsv, {'channel': 'v'}),
        ("hsv_s", pcv.rgb2gray_hsv, {'channel': 's'}),
    ]:
        try:
            g = fn(rgb_img=rgb_img, **kw)
            try:
                g = pcv.transform.rescale(gray_img=g, min_value=0, max_value=255)
                if g.dtype != np.uint8:
                    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            except Exception:
                # ถ้า rescale ล้มเหลว ใช้ค่าเดิม
                pass
            chans.append((nm, g))
        except Exception:
            continue

    if not chans:
        # ไม่มีช่องสีให้ลองเลย → fallback (ยังคง normalize orientation)
        fb = pcv.threshold.otsu(gray_img=pcv.rgb2gray_lab(rgb_img=rgb_img, channel='a'), object_type='dark')
        fb = ensure_binary(fb, normalize_orientation=True)
        h_mm, w_mm = _per_mm_from_shape(height_shape, width_shape)
        return fb, {
            "channel": "lab_a",
            "method": "otsu",
            "object_type": "dark",
            "ksize": None,
            "area_ratio": float(np.count_nonzero(fb)) / max(area_total, 1),
            "Height_shape" :height_shape,
            "Width_shape": width_shape,
            "n_components": 0,
            "solidity": 0.0,
            "Height_mm": h_mm,
            "Width_mm": w_mm,
        }

    # 2) เตรียมชุด threshold ที่จะลอง
    methods = []
    for name, _g in chans:
        methods += [
            (name, ("otsu",     None,  None, "dark")),
            (name, ("otsu",     None,  None, "light")),
            (name, ("triangle", None,  None, "dark")),
            (name, ("triangle", None,  None, "light")),
        ]
        for ksz in (31, 51, 101, 75, 121):
            methods += [
                (name, ("gaussian", ksz, 0, "dark")),
                (name, ("gaussian", ksz, 0, "light")),
            ]

    # 3) ลองทุก candidate + ให้คะแนน + เก็บ metadata
    candidates = []
    for name, spec in methods:
        try:
            gray = next(g for nm, g in chans if nm == name)
            kind, ksz, off, obj_type = spec

            if kind == "otsu":
                m = pcv.threshold.otsu(gray_img=gray, object_type=obj_type)
            elif kind == "triangle":
                m = pcv.threshold.triangle(gray_img=gray, object_type=obj_type)
            else:  # gaussian
                m = pcv.threshold.gaussian(gray_img=gray, ksize=ksz, offset=off, object_type=obj_type)

            m = ensure_binary(m, normalize_orientation=True)

            _fc = cv2.findContours(m.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = _fc[2] if len(_fc) == 3 else _fc[0]
            n_comp = len(contours)
            if n_comp > 0:
                areas = [cv2.contourArea(c) for c in contours]
                cnt = contours[int(np.argmax(areas))]
                hull = cv2.convexHull(cnt)
                a_obj = float(cv2.contourArea(cnt))
                a_hull = float(cv2.contourArea(hull)) if hull is not None else 0.0
                solidity = (a_obj / a_hull) if a_hull > 0 else 0.0
            else:
                solidity = 0.0

            ratio = float(np.count_nonzero(m)) / max(area_total, 1)

            # เกณฑ์แบบ view-aware + ควบคุมด้วย config
            target = float(getattr(cfg, "MASK_TARGET_COVERAGE", 0.05))
            view = str(getattr(cfg, "VIEW", "top")).lower()
            if view == "top":
                exp_min = int(getattr(cfg, "TOP_EXPECT_N_MIN", 5))
                exp_max = int(getattr(cfg, "TOP_EXPECT_N_MAX", 6))
            else:
                exp_min = int(getattr(cfg, "SIDE_EXPECT_N_MIN", 1))
                exp_max = int(getattr(cfg, "SIDE_EXPECT_N_MAX", 1))

            # คะแนนจำนวนก้อน (ให้รางวัลเมื่ออยู่ในช่วงคาดหวัง)
            if n_comp == 0:
                comp_score = -1.0
            elif n_comp < exp_min:
                comp_score = -0.2 * float(exp_min - n_comp)
            elif n_comp > exp_max:
                comp_score = -0.2 * float(n_comp - exp_max)
            else:
                comp_score = +0.3

            # ลดคะแนนเมื่อชนขอบภาพ (border penalty)
            H, W = m.shape[:2]
            top_p    = np.count_nonzero(m[0,  :] > 0) / max(W, 1)
            bottom_p = np.count_nonzero(m[-1, :] > 0) / max(W, 1)
            left_p   = np.count_nonzero(m[:,  0] > 0) / max(H, 1)
            right_p  = np.count_nonzero(m[:, -1] > 0) / max(H, 1)
            n_edges = sum(p > 0.5 for p in (top_p, bottom_p, left_p, right_p))
            edge_strength = top_p + bottom_p + left_p + right_p
            border_penalty = 0.4 * n_edges + 0.6 * edge_strength

            # สรุปคะแนน
            score = (cfg.W_COVERAGE * (1.0 - abs(target - ratio))
            + cfg.W_COMPONENTS * comp_score
            + cfg.W_SOLIDITY * solidity
            - cfg.W_BORDER * border_penalty)

            candidates.append((score, name, m, ratio, n_comp, solidity, kind, obj_type, ksz))
        except Exception:
            continue

    # 4) เลือกตัวที่ดีที่สุด + คืนผล (normalize แล้ว)
    if not candidates:
        fb = pcv.threshold.otsu(gray_img=pcv.rgb2gray_lab(rgb_img=rgb_img, channel='a'), object_type='dark')
        fb = ensure_binary(fb, normalize_orientation=True)
        return fb, {
            "channel": "lab_a",
            "method": "otsu",
            "object_type": "dark",
            "ksize": None,
            "area_ratio": float(np.count_nonzero(fb)) / max(area_total, 1),
            "Height_shape" :height_shape,
            "Width_shape": width_shape,
            "n_components": 0,
            "solidity": 0.0,
        }

    best = max(candidates, key=lambda x: x[0])
    _, name, mask, ratio, n_comp, solidity, kind, obj_type, ksz = best
    m_final = ensure_binary(mask, normalize_orientation=True)
    return m_final, {
        "channel": name,
        "method": kind,
        "object_type": obj_type,
        "ksize": ksz,
        "area_ratio": float(ratio),
        "Height_shape" :height_shape,
        "Width_shape": width_shape,
        "n_components": int(n_comp),
        "solidity": float(solidity),
    }
def _gray_from_channel(rgb_img, ch: str):
    ch = (ch or "").lower()
    if ch in ("lab_a", "lab-a", "laba"):
        return pcv.rgb2gray_lab(rgb_img=rgb_img, channel='a')
    if ch in ("lab_b", "lab-b", "labb"):
        return pcv.rgb2gray_lab(rgb_img=rgb_img, channel='b')
    if ch in ("lab_l", "lab-l", "labl"):
        return pcv.rgb2gray_lab(rgb_img=rgb_img, channel='l')
    if ch in ("hsv_h", "h", "hue"):
        return pcv.rgb2gray_hsv(rgb_img=rgb_img, channel='h')
    if ch in ("hsv_s", "s", "sat"):
        return pcv.rgb2gray_hsv(rgb_img=rgb_img, channel='s')
    if ch in ("hsv_v", "v", "val"):
        return pcv.rgb2gray_hsv(rgb_img=rgb_img, channel='v')
    # ค่าเริ่มต้น: lab_a
    return pcv.rgb2gray_lab(rgb_img=rgb_img, channel='a')

def _summarize_mask(mask_bin):
    m = ensure_binary(mask_bin, normalize_orientation=True)
    H, W = m.shape[:2]
    area_total = max(H*W, 1)
    area_ratio = float(cv2.countNonZero(m)) / area_total

    _fc = cv2.findContours(m.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = _fc[2] if len(_fc) == 3 else _fc[0]
    n_comp = len(contours)
    solidity = 0.0
    if n_comp > 0:
        areas = [cv2.contourArea(c) for c in contours]
        cnt = contours[int(np.argmax(areas))]
        hull = cv2.convexHull(cnt)
        a_obj = float(cv2.contourArea(cnt))
        a_hull = float(cv2.contourArea(hull)) if hull is not None else 0.0
        solidity = (a_obj / a_hull) if a_hull > 0 else 0.0
    return m, area_ratio, n_comp, solidity

def _mask_from_spec(rgb_img, spec: dict):
    ch = spec.get("channel", "lab_a")
    method = (spec.get("method", "otsu") or "otsu").lower()
    obj = spec.get("object_type", "dark") or "dark"
    ksize = spec.get("ksize", None)
    offset = spec.get("offset", 0)
    gray = _gray_from_channel(rgb_img, ch)
    
    if method == "binary":
        thr = int(spec.get("threshold", 128))
        # safety clamp
        if thr < 0: thr = 0
        if thr > 255: thr = 255
        m = pcv.threshold.binary(gray_img=gray, threshold=thr, object_type=obj)

    elif method == "otsu":
        m = pcv.threshold.otsu(gray_img=gray, object_type=obj)

    elif method == "triangle":
        m = pcv.threshold.triangle(gray_img=gray, object_type=obj)
        
    elif method == "mean":
        ksize = int(spec.get("ksize", 51))   # ค่าเริ่มต้น เช่น 51
        offset = int(spec.get("offset", 0))
        m = pcv.threshold.mean(
            gray_img=gray,
            ksize=ksize,
            offset=offset,
            object_type=obj
        )
        
    elif method == "side_auto":
        t, m_guard = auto_thresh_lab_a_otsu_guard(
            rgb_img,
            object_type=spec.get("object_type", "dark"),
            
            thresh_method=spec.get("thresh_method", "fixed"),
            thresh_val=spec.get("thresh_val", 120),
            
            v_bg=spec.get("v_bg", 45),
            s_min=spec.get("s_min", 30),
            v_shadow_max=spec.get("v_shadow_max", 115),
            green_h=spec.get("green_h", (35, 95)),
            use_bottom_roi=True,
            bottom_roi_ratio=spec.get("bottom_roi_ratio", 0.80),
            min_cc_area=spec.get("min_cc_area", 200),
            
            s_min_green=spec.get("s_min_green", 50),
            v_min_green=spec.get("v_min_green", 0),
            )
        m = ensure_binary(m_guard, normalize_orientation=True)

    else:
        # gaussian ต้องมี ksize
        if not ksize:
            raise ValueError("MASK_SPEC requires 'ksize' for gaussian.")
        m = pcv.threshold.gaussian(
            gray_img=gray, ksize=int(ksize), offset=int(offset), object_type=obj
        )
    H, W = rgb_img.shape[:2]      
    height_shape = H
    width_shape = W
    m, area_ratio, n_comp, solidity = _summarize_mask(m)
    info = {
        "source": "manual_spec",
        "channel": str(ch),
        "method": str(method),
        "object_type": str(obj),
        "ksize": int(ksize) if ksize is not None else None,
        "area_ratio": float(area_ratio),
        "Height_shape" :height_shape,
        "Width_shape": width_shape,
        "n_components": int(n_comp),
        "solidity": float(solidity),
    }
    

    if str(getattr(cfg, "VIEW", "top")).lower() == "side":
        bx = int(getattr(cfg, "SIDE_BRIDGE_GAP_X", 100))
        by = int(getattr(cfg, "SIDE_BRIDGE_GAP_Y", 200))
        if bx > 0 or by > 0:
            before_area = cv2.countNonZero(m)
            m = connect_mask_holes(m, gap_x=bx, gap_y=by, iterations=3)
            after_area = cv2.countNonZero(m)
            if getattr(cfg, "DEBUG_MODE", "none") == "print":
                print(f"[bridge-spec] VIEW=side bx={bx} by={by} area {before_area}->{after_area}")
    return m, info

def _mask_from_file(path_str: str):
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"MASK_PATH not found: {p}")
    raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    H, W = raw.shape[:2]       
    height_shape = H
    width_shape = W
    m, area_ratio, n_comp, solidity = _summarize_mask(raw)
    info = {
        "source": "manual_file",
        "channel": "file",
        "method": "file",
        "object_type": "n/a",
        "ksize": None,
        "area_ratio": float(area_ratio),
        "Height_shape" :height_shape,
        "Width_shape": width_shape,
        "n_components": int(n_comp),
        "solidity": float(solidity),
        "mask_path": str(p),
    }
    return m, info

def get_initial_mask(rgb_img):
    # 1) ใช้ไฟล์ mask ถ้ามีกำหนดไว้
    if getattr(cfg, "MASK_PATH", None):
        try:
            return _mask_from_file(cfg.MASK_PATH)
        except Exception as e:
            print("WARN: MASK_PATH failed, fallback to next mode:", e)

    # 2) ใช้ spec threshold ถ้ากำหนด
    if getattr(cfg, "MASK_SPEC", None):
        try:
            return _mask_from_spec(rgb_img, cfg.MASK_SPEC)
        except Exception as e:
            print("WARN: MASK_SPEC failed, fallback to auto:", e)

    # 3) ไม่ได้กำหนด → auto
    m, info = auto_select_mask(rgb_img)
    info = dict(info or {})
    info["source"] = "auto"
    m = ensure_binary(m, normalize_orientation=True)
    if getattr(cfg, "FORCE_OBJECT_WHITE", False):
        if cv2.countNonZero(m) > (m.size // 2):
            m = cv2.bitwise_not(m)
    if str(getattr(cfg, "VIEW", "top")).lower() == "side":
        bx = int(getattr(cfg, "SIDE_BRIDGE_GAP_X", 100))
        by = int(getattr(cfg, "SIDE_BRIDGE_GAP_Y", 200))
        if bx > 0 or by > 0:
            before_area = cv2.countNonZero(m)
            m = connect_mask_holes(m, gap_x=bx, gap_y=by, iterations=1)
            after_area = cv2.countNonZero(m)
            print(f"[DEBUG] bridge gaps (bx={bx}, by={by}) area: {before_area} -> {after_area}")
    return m, info