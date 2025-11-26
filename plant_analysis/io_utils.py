import json
from pathlib import Path
import pandas as pd
from plantcv import plantcv as pcv

# ---------- Utilities ----------
def safe_readimage(path: Path):
    """รองรับต่างเวอร์ชันของ pcv.readimage (2 หรือ 3 ค่าที่คืนมา)"""
    ri = pcv.readimage(filename=str(path))
    img, path_str, filename = None, None, None
    if isinstance(ri, tuple):
        if len(ri) == 3:
            img, path_str, filename = ri
        elif len(ri) == 2:
            img, path_str = ri
            filename = Path(path_str).name if path_str else path.name
        elif len(ri) == 1:
            img = ri[0]
            filename = path.name
        else:
            img = ri[0]; filename = path.name
    else:
        img = ri; filename = path.name
    return img, filename

def aggregate_json_to_csv(json_paths, out_csv: Path):
    rows = []
    for jp in json_paths:
        try:
            with open(jp, 'r', encoding='utf-8') as f:
                rows.append(json.load(f))
        except Exception as e:
            print(f"Error reading {jp}: {e}")
    if not rows:
        print("No valid JSON files found.")
        return
    df = pd.DataFrame(rows)
    cols = list(df.columns)
    
    if 'filename' in cols:
        cols.insert(0, cols.pop(cols.index('filename')))
        df = df[cols]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"Results written to {out_csv}")

def build_file_list(input_path: Path, extensions):
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in extensions else []
    elif input_path.is_dir():
        return [p for p in input_path.glob('**/*') if p.suffix.lower() in extensions]
    else:
        return []

