from flask import Flask, send_file
from flask_cors import CORS
import os
import sys
from pathlib import Path 

app = Flask(__name__)
CORS(app)


RESULTS_FOLDER = r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\result_plant_anomaly"

TOPVIEW_PREFIX = "topview" 
SIDEVIEW_PREFIX = "sideview" 

# ================================================================

def print_error(msg):
    """Helper for printing errors to console"""
    print(f"\n[API SERVER ERROR] ***\n{msg}\n***\n", file=sys.stderr)

def find_latest_file_by_prefix(folder_path, prefix):

    try:
        p = Path(folder_path)
        if not p.is_dir():
            print_error(f"Results folder NOT FOUND or is not a directory:")
            print_error(f"  -> {folder_path}")
            print_error(f"Please check the 'RESULTS_FOLDER' variable in image_api.py")
            return None

        files = list(p.glob(f"*{prefix}*_overlay.jpg"))

        if not files:
            print_error(f"No files found matching '*{prefix}*_overlay.jpg' in {folder_path}")
            return None

        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        return latest_file
        
    except Exception as e:
        print_error(f"Error in find_latest_file: {e}")
        return None


@app.route('/topview_image_result')
def get_topview_image():
    """Endpoint for serving the LATEST Topview result"""
    
    latest_path = find_latest_file_by_prefix(RESULTS_FOLDER, TOPVIEW_PREFIX)
    
    if latest_path is None:
        msg = f"No result file found for prefix: '{TOPVIEW_PREFIX}'"
        return msg, 404
        
    try:
        return send_file(
            latest_path,
            mimetype='image/jpeg',
            max_age=0 
        )
    except Exception as e:
        print_error(f"Error serving topview: {e}")
        return str(e), 500

@app.route('/sideview_image_result')
def get_sideview_image():
    """Endpoint for serving the LATEST Sideview result"""

    latest_path = find_latest_file_by_prefix(RESULTS_FOLDER, SIDEVIEW_PREFIX)

    if latest_path is None:
        msg = f"No result file found for prefix: '{SIDEVIEW_PREFIX}'"
        return msg, 404

    try:
        return send_file(
            latest_path,
            mimetype='image/jpeg',
            max_age=0
        )
    except Exception as e:
        print_error(f"Error serving sideview: {e}")
        return str(e), 500

if __name__ == '__main__':
    print("="*50)
    print(f"Host: 0.0.0.0, Port: 5001")
    print("="*50)
    print(f" Watching FOLDER:\n   {RESULTS_FOLDER}")
    print(f"   - Topview Prefix: '*{TOPVIEW_PREFIX}*_overlay.jpg'")
    print(f"   - Sideview Prefix: '*{SIDEVIEW_PREFIX}*_overlay.jpg'")
    print("="*50)
    
    if not Path(RESULTS_FOLDER).is_dir():
        print(f"*** WARNING: FOLDER NOT FOUND ***")
        print(f"'{RESULTS_FOLDER}'")
        print(f"Make sure this network drive (R:) is connected.")
    else:
        print("Folder found. Waiting for requests from ThingsBoard...")
        
    app.run(host='0.0.0.0', port=5001)