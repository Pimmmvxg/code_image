import cv2, numpy as np

def auto_thresh_lab_a_otsu_guard(
        rgb_img,
        object_type="dark",
        thresh_method="fixed",  # "otsu" หรือ "fixed"
        thresh_val=150,      # ใช้เมื่อ method="fixed"
        # 1) คัดพื้นหลังดำคร่าว ๆ
        v_bg=51,                 # กันฉากดำ/มืด
        blur_ksize=3,
        # 2) กันเงา/พื้นหลังเทา
        s_min=39,                # S ต่ำ = เทา → ตัดทิ้ง
        v_shadow_max=153,        # V ต่ำ = มืด/เงา → ตัดทิ้ง
        v_high_guard=255,
        # 3) บังคับโทน “เขียว”
        green_h=(21, 40),        # ช่วง H ของใบเขียว (OpenCV: 0–179)
        # 4) (ทางเลือก) ใช้ ROI เฉพาะส่วนล่างของภาพ
        use_bottom_roi=True,
        bottom_roi_ratio=0.80,   # ใช้สัดส่วนกี่ % จากล่างขึ้นบน (0.60 = ล่าง 60%)
        # 5) ทำความสะอาด
        min_cc_area=200,         # พื้นที่ขั้นต่ำ
        open_ksize=3,
        close_ksize=8,
        
        s_min_green=10,
        v_min_green=45,
    ):
    H, W, _ = rgb_img.shape

    # === แปลงสี ===
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    Hh, Ss, Vv = cv2.split(hsv)
    L, Aa, Bb = cv2.split(lab)

    if blur_ksize and blur_ksize > 1:
        Aa = cv2.medianBlur(Aa, blur_ksize)

    # === คัดพื้นหลังดำออกคร่าว ๆ (foreground candidate) ===
    fg = (Vv >= v_bg)
    s_prefilter = (Ss >= s_min)
    combined_filter = fg & s_prefilter

    if thresh_method == "side_auto":
        # === คำนวณ Otsu บน a-channel เฉพาะ fg ===
        a_fg = Aa[combined_filter]
        if a_fg.size < 100:
            return 123, np.zeros((H, W), np.uint8)

        hist, _ = np.histogram(a_fg, bins=256, range=(0,255))
        p = hist.astype(np.float64); p /= (p.sum() + 1e-12)
        omega = np.cumsum(p)
        mu = np.cumsum(p * np.arange(256))
        mu_t = mu[-1]
        sigma_b2 = (mu_t*omega - mu)**2 / (omega*(1.0-omega) + 1e-12)
        t = int(np.nanargmax(sigma_b2))
    else: # method == "fixed"
        t = thresh_val

    mask = (Aa <= t) if object_type=="dark" else (Aa >= t)
    mask &= fg

    # === Shadow/Saturation guard ===
    shadow_gray = (Ss <= s_min) & (Vv <= v_shadow_max)  # เทาและมืด -> เงา
    mask &= ~shadow_gray

    # === Green Hue guard ===
    h0, h1 = green_h
    green_band = (Hh >= h0) & (Hh <= h1)
    mask &= green_band
    
    # === (ทางเลือก) ROI เฉพาะส่วนล่างของภาพ ===
    if use_bottom_roi:
        y0 = int((1.0 - bottom_roi_ratio) * H)
        roi = np.zeros((H, W), bool); roi[y0:H, :] = True
        mask &= roi
        
    # === Morphology clean-up ===
    mask_u8 = (mask.astype(np.uint8) * 255)
    if open_ksize > 1:
        kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel_o, iterations=1)
    if close_ksize > 1:
        kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_c, iterations=2)

    # ตัดชิ้นส่วนเล็ก ๆ
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros_like(mask_u8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_cc_area:
            cleaned[labels == i] = 255
            
    s_min_green = 10
    v_min_green = 0
    green_sv_gaurd = (Ss >= s_min_green) & (Vv >= v_min_green)
    cleaned = cv2.bitwise_and(cleaned, green_sv_gaurd.astype(np.uint8)*255)

    return t, cleaned