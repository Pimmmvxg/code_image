import json, time, math
from pathlib import Path
from .tb_client import ThingsboardClient

def publish_data(json_path: str | Path,
                 telemetry_keys = ("side_1_height_mm", "side_1_length_mm",
                                  "side_2_height_mm", "side_2_length_mm",
                                  "global_color_name", 
                                  "top_1_color_name", "top_1_area_mm2",
                                  "top_2_color_name", "top_2_area_mm2",
                                  "top_3_color_name", "top_3_area_mm2",
                                  "top_4_color_name", "top_4_area_mm2",
                                  "top_5_color_name", "top_5_area_mm2",
                                  "top_6_color_name", "top_6_area_mm2",
                                  "top_7_color_name", "top_7_area_mm2",
                                  "top_8_color_name", "top_8_area_mm2",
                                  "top_color_name", "top_area_mm2",
                                  "filename"
                                  ),
                 ):
    p = Path(json_path)
    if not p.is_file():
        raise FileNotFoundError(f"Result JSON not found: {p}")
    
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    telemetry = {}
    missing = []
    for key in telemetry_keys:
        if key in data:
            val = data[key]
            # กัน NaN/Inf
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                continue
            telemetry[key] = val
        else:
            missing.append(key)

    if not telemetry:
        raise ValueError(f"No telemetry keys found in JSON. Expected any of: {telemetry_keys}")

    # 4) ส่งขึ้น ThingsBoard
    cli = ThingsboardClient()
    try:
        cli.publish_attributes(telemetry)
    finally:
        try:
            cli.stop()
        except Exception:
            pass

    return {"telemetry": telemetry, "missing": missing, "json_path": str(p)}