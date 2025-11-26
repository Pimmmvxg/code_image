#convert hue to color name
def _hue_to_degrees(h):
    """Normalize various hue scales to [0, 360). Accepts python or numpy scalars."""
    import numpy as np
    # cast to float
    h = float(np.asarray(h).item())  # handles numpy types safely
    # heuristic: if clearly in OpenCV 0–179, scale x2
    if 0.0 <= h <= 179.0:
        h *= 2.0
    # if looks like 0–1, scale x360
    if 0.0 <= h <= 1.0:
        h *= 360.0
    # wrap
    h = (h % 360.0 + 360.0) % 360.0
    return h

def get_color_name(hue):
    if hue < 15 or hue >= 345:
        return "Red"
    elif hue < 30:
        return "Orange"
    elif hue < 45:
        return "Warm Yellow"
    elif hue < 60:
        return "Mid Yellow"
    elif hue < 75:
        return "Cool Yellow"
    elif hue < 90:
        return "Yellow Green"
    elif hue < 105:
        return "Warm Green"
    elif hue < 120:
        return "Mid Green"
    elif hue < 135:
        return "Cool Green"
    elif hue < 150:
        return "Green Cyan"
    elif hue < 195:
        return "Cool Cyan"
    elif hue < 225:
        return "Blue"
    elif hue < 255:
        return "Deep Blue"
    elif hue < 285:
        return "Blue violet"
    elif hue < 315:
        return "violet"
    else:
        return 'Unknown'
    
