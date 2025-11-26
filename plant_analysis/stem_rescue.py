import cv2, numpy as np, time

def add_v_connected_to_a(
    rgb, base_a_mask,
    # ทำ v-mask
    method="fixed",            # "fixed" | "otsu" | "percentile"
    v_min=150, v_max=255,      # ใช้เมื่อ method="fixed"
    percentile=85,             # ใช้เมื่อ method="percentile"
    s_max=None,                # ใส่ค่านี้ถ้าต้องการกันของสีซีด (S ต่ำ) เช่น 80
    # กันแฟลช/สเกลขาวจัด
    glare_v=255, glare_s=0,   # V>=glare_v และ S<=glare_s จะถูกตัดทิ้งจาก v-mask
    # ขอบเขตการเชื่อม
    near_px=20,                # บัฟเฟอร์รอบฐาน (ช่วยให้ “แตะ” กันได้)
    geo_iters=100,             # รอบ geodesic (60–120 ถ้าก้านยาว)
    # ความสะอาด
    open_k=3,                  # เปิดรูเล็กบน v-mask
    min_area_keep=400,         # ตัดคอมโพเนนต์เล็กหลังเชื่อมแล้ว
    
    connect_mode="geo",
    cc_close_k=0
):
    t0 = time.time()
    it = None
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H,S,V = cv2.split(hsv)

    # ---------- 1) สร้าง v-mask ----------
    if method == "fixed":
        _, vmask = cv2.threshold(V, v_min, v_max, cv2.THRESH_BINARY)
        thr_note = f"plantcv_light: V>{int(v_min)}"
        # (ถ้าต้องการ upper bound ด้วย v_max ให้ AND เพิ่ม)
        if v_max is not None and v_max < 255:
            vmask = cv2.bitwise_and(vmask, cv2.inRange(V, 0, int(v_max)))
    else:
        # ใช้ข้อมูลเฉพาะ "ใกล้ฐาน" เพื่อให้ threshold เสถียรกว่า
        near = cv2.dilate(base_a_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(near_px,near_px)), 1)
        vals = V[near>0]
        if method == "otsu":
            vv = V.copy(); vv[near==0] = 0
            _, t = cv2.threshold(vv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            vmask = cv2.inRange(V, int(t), v_max if v_max is not None else 255)
            thr_note = int(t)
        else:  # percentile
            t = int(np.percentile(vals, percentile)) if len(vals) else 200
            vmask = cv2.inRange(V, t, v_max if v_max is not None else 255)
            thr_note = int(t)

    if s_max is not None:
        vmask = cv2.bitwise_and(vmask, (S <= int(s_max)).astype(np.uint8)*255)

    # กันแฟลช/สเกลขาวจัด
    if glare_v is not None and glare_s is not None:
        glare = ((V >= int(glare_v)) & (S <= int(glare_s))).astype(np.uint8)*255
        vmask = cv2.bitwise_and(vmask, cv2.bitwise_not(glare))

    # เปิดรูเล็ก
    if open_k and open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        vmask = cv2.morphologyEx(vmask, cv2.MORPH_OPEN, k, iterations=1)

    # ---------- 2) เชื่อมเฉพาะส่วนที่ ติดกับฐานเดิม(lab a) ----------
    if str(connect_mode).lower() == "cc_touch":
        # เก็บทั้งก้อน ของ vmask ที่แตะขอบฐาน (dilate ด้วย near_px)
        near = (base_a_mask > 0).astype(np.uint8)
        if near_px and int(near_px) > 0:
            knear = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(near_px), int(near_px)))
            near = cv2.dilate(near, knear, 1)

        vsrc = vmask.copy()
        try:
            gap_x = int(1)
            gap_y = int(1)
            iters = 3
            if gap_x > 0:
                h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2* gap_x +1, 3))
                vsrc = cv2.morphologyEx(vsrc, cv2.MORPH_CLOSE, h_kernel, iterations=iters)
            if gap_y > 0:
                v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2* gap_y +1))
                vsrc = cv2.morphologyEx(vsrc, cv2.MORPH_CLOSE, v_kernel, iterations=iters)
        except Exception as e:
            print(f"[stem-rescue] side-view bridge error: {e}")
        num, lbl, stats, _ = cv2.connectedComponentsWithStats((vsrc>0).astype(np.uint8), 8)
        
        keep = np.zeros_like(vsrc, np.uint8)

        if num > 1:
            # label ของพิกเซลที่แตะฐาน
            touch_ids = np.unique(lbl[near > 0])
            for i in touch_ids:
                if i <= 0:
                    continue
                if min_area_keep and stats[i, cv2.CC_STAT_AREA] < int(min_area_keep):
                    continue
                keep[lbl == i] = 255

        v_connected = keep

    else:
        # โหมดเดิม: geodesic dilation คืบทีละพิกเซลใน vmask
        seeds = (base_a_mask > 0).astype(np.uint8)*255
        mask  = vmask
        prev  = np.zeros_like(seeds)
        se3   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        it = 0
        while it < int(geo_iters):
            it += 1
            seeds = cv2.dilate(seeds, se3, iterations=1)
            seeds = cv2.bitwise_and(seeds, mask)
            if np.array_equal(seeds, prev): break
            prev = seeds.copy()
        v_connected = seeds  # ทุกอันจาก V ที่เชื่อมต่อกับฐาน a

    # ---------- 3) ตัดคอมโพเนนต์เล็ก ๆ ----------
    if min_area_keep and min_area_keep > 0:
        n, lbl, stats, _ = cv2.connectedComponentsWithStats((v_connected>0).astype(np.uint8), 8)
        kept = np.zeros_like(v_connected)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= int(min_area_keep):
                kept[lbl==i] = 255
        v_connected = kept

    # ---------- 4) รวมกับฐาน ----------
    final = cv2.bitwise_or(base_a_mask, v_connected)

    debug = {
        "V": V, "S": S, "vmask": vmask, "v_connected": v_connected, "final": final,
        "thr_note": thr_note, "iters_done": it, "timing_ms": int((time.time()-t0)*1000),
    }
    return final, debug
