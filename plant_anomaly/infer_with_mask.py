"""
YOLO-Seg inference with background removal by mask (pre-mask) + optional tiling.
- ตัดพื้นหลังด้วยมาสก์ (AND + optional tight-crop)
- (Option) แบ่งเป็นไทล์ต่อพุ่มจาก connected components แล้วค่อย infer
- บังคับผลทำนายให้อยู่ในมาสก์ (post-process) + ผ่อนเกณฑ์สำหรับรอยเล็กริมใบ
- เซฟ overlay/ผลลัพธ์ JSON/ดีบักเปรียบเทียบ
ทดสอบแล้วกับ Ultralytics YOLO v8/9 (seg)
"""

from __future__ import annotations
import os, re, json, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from numpy import ndarray

# ============ CONFIG ============
CFG = {
    # ---- Paths ----
    #"weights": r"C:\leaf_diseases\yolo-seg\runs\seg_train\exp\weights\best.pt",
    "weights" : r"C:\Plant_analysis\plant_anomaly\best.pt",
    "img_path": r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_topview_smartfarm\picture_topview_B_12112025_100120.jpg",      # รูปเดี่ยวหรือโฟลเดอร์
    "mask_dir": None , # โฟลเดอร์มาสก์ (None ถ้าไม่มี)
    #"out_dir":  r"C:\leaf_diseases\yolo-seg\results_infer",

    # ---- Inference ----
    "imgsz": 1024,           # เพิ่มเพื่อเห็นรอยเล็ก
    "conf": 0.3,            # ค่าความมั่นใจขั้นต่ำ
    "iou": 0.6,
    "max_det": 500,
    "device": 0,             # 0=GPU ตัวแรก, "cpu"=ซีพียู
    "half": False,           # ใช้ True เมื่อ GPU รองรับ FP16
    "augment": False,         # ช่วย recall

    # ---- Pre-mask options ----
    "use_sidecar_mask": True,
    "skip_if_no_mask": False,
    "mask_binarize_thresh": 127,
    "mask_dilate": 3,        # 2–5 ช่วยไม่ให้รอยริมใบหาย
    "tight_crop": True,
    "crop_pad_px": 25,
    "auto_invert_if_cover_gt": None,  # 0.7 ถ้ามาสก์สีตรงข้าม (คลุมพื้นที่ >70%) จะ invert ให้อัตโนมัติ
    
    #draw
    "draw_boxes": False,
    "draw_mask_style": "contour",  # "contour" หรือ "filled"

    # ---- Tile by connected components (ต่อพุ่ม/กลุ่มใบ) ----
    "use_tiling": True,
    "tile_min_area": 2000,
    "tile_pad": 25,

    # ---- Post-process: constrain prediction to mask ----
    "min_overlap": 0.30,     
    "min_area_px": 0,       # เก็บรอยเล็ก
    # ถ้าจะเปรียบเทียบก่อน vs หลังตัดมาสก์ ให้เปิด:
    "debug_compare": True,

    # ---- Saving ----
    "save_overlay": True,
    "save_masked_image": True,
    "save_json": True,

}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============ Utils ============

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def list_images(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in IMG_EXTS: return [path]
    out = []
    for ext in IMG_EXTS: out.extend(path.rglob(f"*{ext}"))
    return out

def find_mask_for_image(img_path: Path, mask_dir: Optional[Path]) -> Optional[Path]:
    if not mask_dir or not mask_dir.exists(): return None
    stem = img_path.stem
    cands = [f"{stem}{e}" for e in [".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"]]
    cands += [f"{stem}_mask{e}" for e in [".png",".jpg",".jpeg",".webp"]]
    cands += [f"{stem}-mask{e}" for e in [".png",".jpg",".jpeg",".webp"]]
    for c in cands:
        hit = list(mask_dir.rglob(c))
        if hit: return hit[0]
    pat = re.compile(re.escape(stem) + r".*mask", re.IGNORECASE)
    for p in mask_dir.rglob("*"):
        if p.is_file() and pat.search(p.stem): return p
    return None

def make_binary_mask(mask_img, size_wh, thresh=127, dilate_iter=0, auto_invert_cover=None):
    # to gray
    if mask_img.ndim == 3:
        c = mask_img.shape[2]
        if c == 4: gray = mask_img[:, :, 3]
        elif c == 3: gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        elif c == 1: gray = mask_img[:, :, 0]
        else: gray = np.squeeze(mask_img)
    else:
        gray = mask_img
    gray = np.ascontiguousarray(gray)

    if gray.dtype != np.uint8:
        g = gray.astype(np.float32)
        rng = float(g.max() - g.min())
        gray = (np.zeros_like(g) if rng < 1e-6 else (g - g.min()) * (255.0 / rng)).astype(np.uint8)

    w, h = size_wh
    gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)

    if dilate_iter and dilate_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.dilate(m, k, iterations=int(dilate_iter))
    return m

def apply_mask_and_crop(img: np.ndarray, mask: np.ndarray,
                        tight_crop=True, pad_px=20) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """AND mask กับภาพ แล้ว (Option) crop เฉพาะบริเวณ mask"""
    assert img.shape[:2]==mask.shape[:2]
    masked = cv2.bitwise_and(img, img, mask=mask)
    ys, xs = np.where(mask>0)
    if xs.size==0 or ys.size==0:
        h,w = img.shape[:2]; return masked, (0,0,w,h)
    x0,x1 = int(xs.min()), int(xs.max())
    y0,y1 = int(ys.min()), int(ys.max())
    if tight_crop:
        x0 = max(0, x0-pad_px); y0 = max(0, y0-pad_px)
        x1 = min(img.shape[1], x1+pad_px); y1 = min(img.shape[0], y1+pad_px)
        masked = masked[y0:y1, x0:x1]
    return masked, (x0,y0,x1,y1)

def split_tiles_by_components(pre_img: np.ndarray, fg_mask_crop: np.ndarray,
                              min_area=2000, pad=20) -> List[Tuple[np.ndarray,np.ndarray,Tuple[int,int,int,int]]]:
    """หา connected components ในมาสก์ แล้วคืนไทล์ [(img_tile, mask_tile, (x0,y0,x1,y1))]"""
    m = (fg_mask_crop>0).astype(np.uint8)
    num, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    tiles = []
    H,W = m.shape
    for i in range(1, num):
        x,y,w,h,a = stats[i]
        if a < min_area: continue
        x0 = max(0, x-pad); y0 = max(0, y-pad)
        x1 = min(W, x+w+pad); y1 = min(H, y+h+pad)
        tiles.append(( pre_img[y0:y1, x0:x1].copy(), (m[y0:y1, x0:x1]*255), (x0,y0,x1,y1) ))
    if not tiles:
        tiles = [(pre_img, m*255, (0,0,pre_img.shape[1], pre_img.shape[0]))]
    return tiles

def draw_polygons_overlay(base: np.ndarray, results) -> np.ndarray:
    vis = base.copy()
    if not results or len(results) == 0: return vis
    r0 = results[0]
    # ----- วาดจาก masks.data เสมอ เพื่อให้พิกัดตรงหลังกรอง -----
    if getattr(r0, "masks", None) is not None and getattr(r0.masks, "data", None) is not None:
        md = (r0.masks.data > 0.35).detach().cpu().numpy().astype(np.uint8)  # ลด threshold ช่วยรอยบาง/ริมใบ
        oh, ow = r0.masks.orig_shape if hasattr(r0.masks, "orig_shape") else vis.shape[:2]
        for i in range(md.shape[0]):
            m = cv2.resize(md[i], (ow, oh), interpolation=cv2.INTER_NEAREST) * 255
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if CFG.get("draw_mask_style", "contour") == "filled":
                overlay = vis.copy()
                cv2.drawContours(overlay, cnts, -1, (0,0,255), thickness=cv2.FILLED)
                cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
            # เส้นคอนทัวร์
            cv2.drawContours(vis, cnts, -1, (0, 0, 255), 5)

    # ----- (Optional) วาด bbox เฉพาะถ้าเปิด -----
    if CFG.get("draw_boxes", False) and getattr(r0, "boxes", None) is not None and r0.boxes is not None:
        boxes = r0.boxes.xyxy.detach().cpu().numpy()
        confs = r0.boxes.conf.detach().cpu().numpy() if r0.boxes.conf is not None else None
        names = r0.names if hasattr(r0, "names") else {}
        clss = r0.boxes.cls.detach().cpu().numpy().astype(int) if r0.boxes.cls is not None else None
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 220, 0), 2)
            label = ""
            if clss is not None:
                cname = names.get(int(clss[i]), str(int(clss[i]))) if isinstance(names, dict) else str(int(clss[i]))
                label = cname
            if confs is not None:
                label = f"{label} {confs[i]:.2f}" if label else f"{confs[i]:.2f}"
            if label:
                cv2.putText(vis, label, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2, cv2.LINE_AA)
    return vis

def constrain_results_to_mask(r0, fg_mask_crop: np.ndarray, min_overlap=0.3, min_area_px=30):
    if r0 is None:
        return None
    
    if fg_mask_crop is None:
        if not CFG.get("draw_boxes", False) and getattr(r0,"boxes",None) is not None:
            r0.boxes = None
        return r0
    
    H, W = fg_mask_crop.shape[:2]
    
    if getattr(r0,"masks",None) is None or getattr(r0.masks,"data",None) is None:
        if getattr(r0, "boxes", None) is not None and r0.boxes is not None:
            boxes = r0.boxes.xyxy.detach().cpu().numpy().astype(int)
            keep = []
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(W, x2); y2 = min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                roi = fg_mask_crop[y1:y2, x1:x2]
                if np.count_nonzero(roi) > 0:
                    keep.append(i)
            if keep:
                idx = torch.as_tensor(keep, device=r0.boxes.data.device)
                r0.boxes = r0.boxes[idx]
            else:
                r0.boxes = None
        
        if not CFG.get("draw_boxes", False) and getattr(r0,"boxes",None) is not None:
            r0.boxes = None
        return r0
    
    md = (r0.masks.data > 0.35).detach().cpu().numpy().astype(np.uint8)
    if hasattr(r0.masks, "orig_shape") and r0.masks.orig_shape is not None:
        oh, ow = r0.masks.orig_shape
    else:
        oh, ow = H, W
    
    kept_masks = []
    kept_indices = []
    areas = []
    
    fg = (fg_mask_crop>0).astype(np.uint8)
    
    for i in range(md.shape[0]):
        m = cv2.resize(md[i], (ow, oh), interpolation=cv2.INTER_NEAREST)
        inter = (m & fg).astype(np.uint8)
        
        area_m = int(m.sum())
        area_i = int(inter.sum())
        
        overlap = (area_i / (area_m + 1e-6)) if area_m > 0 else 0.0
        if area_i >= max(1, min_area_px) and overlap >= float(min_overlap):
            kept_masks.append(inter)
            kept_indices.append(i)
            areas.append(area_i)
            
    if len(kept_masks) == 0:
        r0.masks.data = r0.masks.data[:0]
        r0.boxes = None if not CFG.get("draw_boxes", False) else r0.boxes
        return r0
    
    kept_masks_np = np.stack(kept_masks, axis=0).astype(np.uint8)
    kept_tensor = torch.from_numpy((kept_masks_np > 0).astype(np.float32)).to(r0.masks.data.device)
    r0.masks.data = kept_tensor
    if hasattr(r0.masks, "orig_shape"):
        r0.masks.orig_shape = (oh, ow)
        
    new_boxes = []
    for i in range(kept_masks_np.shape[0]):
        m255 = kept_masks_np[i] * 255
        cnts, _ = cv2.findContours(m255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            new_boxes.append([0, 0, 0, 0])
            continue
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        new_boxes.append([x, y, x + w, y + h])
        
    if CFG.get("draw_boxes", False):
        from ultralytics.yolo.engine.results import Boxes
        boxes_xyxy = torch.tensor(new_boxes, dtype=torch.float32, device=r0.masks.data.device)
        if getattr(r0, "boxes", None) is None:
            r0.boxes = Boxes(boxes_xyxy, r0.orig_img if hasattr(r0, "orig_img") else None)
        else:
            try:
                idx = torch.as_tensor(kept_indices, device=r0.boxes.data.device)
                r0.boxes = r0.boxes[idx]
                r0.boxes.xyxy = boxes_xyxy
            except Exception:
                r0.boxes.xyxy = boxes_xyxy
    else:
        r0.boxes = None
    
    return r0

# ============ Internal Pipeline Helpers ===========
def _load_image_and_mask(
    image_path: Path,
    mask_input: Optional[Union[Path, np.ndarray]],
    cfg: dict
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    mask_bin = None
    used_mask_input = None
    
    if cfg["use_sidecar_mask"]:
        if mask_input is None and cfg.get("mask_dir"):
            md = Path(cfg["mask_dir"])
            if md.exists():
                used_mask_input = find_mask_for_image(image_path, md) if md.is_dir() else md
                
        else:
            used_mask_input = mask_input
    
    #ประมวลผล mask input
    if used_mask_input is not None:
        m_img = None
        if isinstance(used_mask_input, Path) and used_mask_input.exists():
            m_img = cv2.imread(str(used_mask_input), cv2.IMREAD_UNCHANGED)
        elif isinstance(used_mask_input, np.ndarray):
            m_img = used_mask_input
        
        if m_img is not None:
            mask_bin = make_binary_mask(
                m_img, (img.shape[1], img.shape[0]),
                thresh=cfg["mask_binarize_thresh"],
                dilate_iter=cfg["mask_dilate"],
                auto_invert_cover=cfg["auto_invert_if_cover_gt"]
            )
        elif cfg["skip_if_no_mask"]:
            raise FileNotFoundError("Mask not found and skip_if_no_mask=True")
        
    return img, mask_bin, used_mask_input
    
def _apply_preprocessing(
    img: np.ndarray,
    mask_bin: Optional[np.ndarray],
    cfg: dict
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Optional[np.ndarray]]:
    
    if mask_bin is None:
        pre_img = img
        crop_box = (0, 0, img.shape[1], img.shape[0])
        fg_mask_crop = None
    else:
        pre_img, crop_box = apply_mask_and_crop(
            img, mask_bin, tight_crop=cfg["tight_crop"], pad_px=cfg["crop_pad_px"]
        )
        x0, y0, x1, y1 = crop_box
        fg_mask_crop = mask_bin[y0:y1, x0:x1] if cfg["tight_crop"] else mask_bin
    
    if cfg["save_masked_image"]:
        mask_status = 'masked' if mask_bin is not None else 'orig'
        pass
    return pre_img, crop_box, fg_mask_crop

def _get_tiles(
    pre_img: np.ndarray,
    fg_mask_crop: Optional[np.ndarray],
    cfg: dict,
    original_crop_box: Tuple[int, int, int, int]
) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], Tuple[int,int]]]:
    
    offset_x, offset_y, _, _ = original_crop_box
    
    final_tiles = []
    
    if cfg["use_tiling"] and fg_mask_crop is not None:
        for tile_img, tile_mask, (tx0, ty0, tx1, ty1) in split_tiles_by_components(
            pre_img, fg_mask_crop,
            cfg["tile_min_area"],
            cfg["tile_pad"]
        ):
            final_tiles.append((tile_img, tile_mask, (tx0, ty0, tx1, ty1), (offset_x, offset_y)))
    else:
        mask = fg_mask_crop if fg_mask_crop is not None else np.ones(pre_img.shape[:2], np.uint8) * 255
        final_tiles.append((pre_img, mask, (0, 0, pre_img.shape[1], pre_img.shape[0]), (offset_x, offset_y)))
    
    return final_tiles

def _run_inference_on_tile(
    model: YOLO,
    tiles: List[tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]],
    vis_full: np.ndarray,
    cfg: dict,
) -> Tuple[np.ndarray, Dict]:
    total_det= 0
    classes, confs = [], []
    
    debug_compare_paths = []
    
    for (tile_img, tile_mask, (tx0_global, ty0_global, tx1_global, ty1_global), (offset_x, offset_y)) in tiles:
        if tile_img.shape[0] == 0 or tile_img.shape[1] == 0:
            continue
        
        tx0_global = tx0_global + offset_x
        ty0_global = ty0_global + offset_y
        tx1_global = tx1_global + offset_x
        ty1_global = ty1_global + offset_y
        
        # 1) predict
        res_raw = model.predict(
            source=tile_img, imgsz=cfg["imgsz"], conf=cfg["conf"], iou=cfg["iou"],
            max_det=cfg["max_det"], device=cfg["device"], half=cfg["half"],
            augment=cfg["augment"], verbose=False
        )
        r0 = res_raw[0] if res_raw else None
        
        # 2) intersect-only เพื่อให้ข้างนอกมาสก์หายไปก่อน (ยังไม่คัดทิ้ง)
        if tile_mask is not None and getattr(r0,"masks",None) is not None and getattr(r0.masks,"data",None) is not None:
            md = r0.masks.data
            if md.dtype!=torch.bool: md = md>0.5
            H,W = md.shape[1:]
            fg = tile_mask
            if (fg.shape[0],fg.shape[1])!=(H,W): fg = cv2.resize(fg, (W,H), interpolation=cv2.INTER_NEAREST)
            fg_t = torch.from_numpy((fg>0).astype(np.bool_)).to(md.device)
            new_masks, new_boxes, keep_idx = [], [], []
            for i in range(md.shape[0]):
                inter = (md[i]) & fg_t
                if inter.sum().item()>0:
                    new_masks.append(inter.unsqueeze(0))
                    inter_np = inter.detach().cpu().numpy().astype(np.uint8)*255
                    ys,xs = np.where(inter_np>0)
                    x1,y1,x2,y2 = int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())
                    new_boxes.append([x1,y1,x2,y2]); keep_idx.append(i)
            if new_masks:
                r0.masks.data = torch.cat(new_masks, dim=0)
                if getattr(r0,"boxes",None) is not None and r0.boxes is not None:
                    device = r0.boxes.data.device
                    r0.boxes = r0.boxes[torch.tensor(keep_idx, device=device)]
                    xyxy_t = torch.tensor(new_boxes, dtype=r0.boxes.data.dtype, device=device)
                    nb = xyxy_t.shape[0]
                    r0.boxes.data = r0.boxes.data.clone()
                    r0.boxes.data[:nb, 0:4] = xyxy_t
            else:
                r0.masks=None; r0.boxes=None

        # 3) กรองจริงตามเกณฑ์ overlap/area
        r0 = constrain_results_to_mask(r0, tile_mask, cfg["min_overlap"], cfg["min_area_px"])
        res_final = [r0] if r0 else []
        
        # 4) วาด overlay คืนภาพเต็ม
        overlay = draw_polygons_overlay(tile_img, res_final)
        
        #ป้องกันขนาดไม่เท่ากัน
        h, w = overlay.shape[:2]
        vis_full_roi = vis_full[ty0_global : ty0_global + h, tx0_global : tx0_global + w]
        if vis_full_roi.shape[:2] == (h, w):
            vis_full[ty0_global : ty0_global + h, tx0_global : tx0_global + w] = overlay
        else:
            #ถ้าขนาดไม่เท่ากัน resize
            vis_full[ty0_global : ty1_global, tx0_global : tx1_global] = cv2.resize(overlay, (vis_full_roi.shape[1], vis_full_roi.shape[0]))
            
        # 5) summary
        if r0 and getattr(r0, "masks", None) is not None and getattr(r0.masks, "data", None) is not None and len(r0.masks.data) > 0:
            num_found = int(r0.masks.data.shape[0])
            total_det += num_found
            
            nm = r0.names if hasattr(r0, "names") else {}
            cl, cf = [], []
            
            if getattr(r0, "boxes", None) is not None and r0.boxes is not None and len(r0.boxes) == num_found:
                cl = r0.boxes.cls.detach().cpu().numpy().astype(int) if r0.boxes.cls is not None else []
                cf = r0.boxes.conf.detach().cpu().numpy().round(4).tolist() if r0.boxes.conf is not None else []
            else:
                cl = [0] * num_found
                cf = [0.0] * num_found
            for i in range(num_found):
                c = cl[i] if i < len(cl) else 0
                c_val = cf[i] if i < len(cf) else 0.0
                classes.append(nm.get(int(c), str(int(c))) if isinstance(nm,dict) else str(int(c)))
                confs.append(c_val)
        
        # 6) debug: ภาพเต็ม vs หลังตัดมาสก์
        if cfg["debug_compare"]:
            try:
                raw_vis = res_raw[0].plot()
                out_path = f"debug_tile_{tx0_global}_{ty0_global}.jpg"
                debug_compare_paths.append((out_path, raw_vis))
            except Exception: pass
            
    summary_data = {
        "num_detections": total_det,
        "classes": classes,
        "confs": confs,
        "debug_compare_paths": debug_compare_paths
    }
    return vis_full, summary_data

def _save_outputs(
    image_path: Path,
    out_dir: Path,
    vis_full: np.ndarray,
    pre_img: np.ndarray,
    mask_bin: Optional[np.ndarray],
    crop_box: Tuple[int,int,int,int],
    used_mask_input: Optional[Union[Path, np.ndarray]],
    summary_data: dict,
    elapsed: float,
    cfg: dict
):
    
    stem = image_path.stem
    
    if cfg["save_overlay"]:
        cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), vis_full)
        
    if cfg["save_masked_image"]:
        status = 'masked' if mask_bin is not None else 'orig'
        cv2.imwrite(str(out_dir / (f"{stem}_{status}.png")), pre_img)
        
    if cfg["debug_compare"]:
        for (rel_path, img_data) in summary_data["debug_compare_paths"]:
            cv2.imwrite(str(out_dir / f"{stem}_{rel_path}"), img_data)
            
        if mask_bin is not None:
            cv2.imwrite(str(out_dir / f"{stem}_mask_used.png"), mask_bin)
            
    mask_str = None
    if used_mask_input is not None:
        mask_str = str(used_mask_input) if isinstance(used_mask_input, Path) else "ndarray_input"
        
    info = {
        "image": str(image_path),
        "used_mask": mask_str,
        "crop_box": crop_box,
        "num_detections": summary_data["num_detections"],
        "classes": summary_data["classes"],
        "confs": summary_data["confs"],
        "elapsed_sec": elapsed
    }
    
    if cfg["save_json"]:
        with open(out_dir / f"{stem}_result.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    
    return info
        
# ============ Core ============

def infer_one(
    model: YOLO,
    cfg: dict,
    image_path: Path,
    out_dir: Path,
    mask_input: Optional[Union[Path, np.ndarray]] = None
) -> Dict:
    
    t0 = time.time()
    
    try:
        # โหลดภาพ และเตรียมมาสก์
        img, mask_bin, used_mask_input = _load_image_and_mask(
            image_path, mask_input, cfg
        )
        
        # ครอปภาพ
        pre_img, crop_box, fg_mask_crop = _apply_preprocessing(
            img, mask_bin, cfg
        )
        
        # แบ่ง tile
        tiles = _get_tiles(pre_img, fg_mask_crop, cfg, crop_box)
        
        # run model, กรอง, วาดผล
        vis_full, summary_data = _run_inference_on_tile(
            model, tiles, img.copy(), cfg
        )
        
        # บันทึกผลลัพธ์
        elapsed = round(time.time() - t0, 4)
        final_summary = _save_outputs(
            image_path, out_dir, vis_full, pre_img,
            mask_bin, crop_box, used_mask_input,
            summary_data, elapsed, cfg
        )
        
        return final_summary
    
    except FileNotFoundError as e:
        print(f"Skipping {image_path.name}: {e}")
        return {"image": str(image_path), "skipped": True, "reason": str(e)}
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return {"image": str(image_path), "error": str(e)}

def infer_dir(cfg=CFG, img_dir: str|Path=None, mask_dir: str|Path|None=None, out_dir: str|Path=None) -> List[Dict]:
    img_dir = Path(img_dir or cfg["img_path"])
    out_dir = Path(out_dir or cfg["out_dir"]); ensure_dir(out_dir)
    mask_dir = Path(mask_dir) if mask_dir is not None else None

    model = YOLO(cfg["weights"])
    # if forcing single class name
    if cfg.get("force_class_names"): model.names = cfg["force_class_names"]

    images = list_images(img_dir)
    summaries = []
    for i, ip in enumerate(images, 1):
        mp = None
        if cfg["use_sidecar_mask"]:
            mp = find_mask_for_image(ip, mask_dir)
        s = infer_one(model, cfg, ip, out_dir, mask_input=mp)
            
        print(f"[{i}/{len(images)}] {ip.name} -> det={s.get('num_detections', 'N/A')} mask={'Y' if s.get('used_mask') else 'N'}")
        summaries.append(s)
            
    if cfg["save_json"]:
        with open(out_dir / "_batch_summary.json", "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
    return summaries


def main():
    out_dir = Path(CFG["out_dir"]); ensure_dir(out_dir)
    p = Path(CFG["img_path"])
    if p.is_dir():
        infer_dir(CFG, p, CFG["mask_dir"], out_dir)
    else:
        model = YOLO(CFG["weights"])
        if CFG.get("force_class_names"): model.names = CFG["force_class_names"]
        
        mask_path = None
        if CFG["use_sidecar_mask"] and CFG.get("mask_dir"):
            mask_path = find_mask_for_image(p, Path(CFG["mask_dir"]))
            
        s = infer_one(model, CFG, p, out_dir, mask_input=mask_path)
            
        print(s)        

if __name__ == "__main__":
    main()
