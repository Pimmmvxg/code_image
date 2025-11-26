import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from . import config as cfg
from .io_utils import build_file_list, aggregate_json_to_csv
from .pipeline import process_one

def main():
    in_path = Path(cfg.INPUT_PATH)
    out_dir = Path(cfg.OUTPUT_DIR)
    exts = set([e.lower() for e in cfg.EXTENSIONS])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "debug").mkdir(exist_ok=True)

    files = build_file_list(in_path, exts)
    if not files:
        print(f"No files found in {in_path} with extensions {exts}.")
        sys.exit(1)

    print(f"Found {len(files)} files to process.")

    json_paths = []
    if cfg.THREADS > 1:
        with ProcessPoolExecutor(max_workers=cfg.THREADS) as ex:
            futures = {ex.submit(process_one, f, out_dir): f for f in files}
            for fut in as_completed(futures):
                f = futures[fut]
                try:
                    jp = fut.result()
                    json_paths.append(jp)
                except Exception as e:
                    print(f"Error processing {f}: {e}")
    else:
        for f in files:
            try:
                jp = process_one(f, out_dir)
                json_paths.append(jp)
            except Exception as e:
                print(f"Error processing {f}: {e}")

    out_csv = out_dir / "results.csv"
    aggregate_json_to_csv(json_paths, out_csv)
    print(f"Results written to {out_csv}")

if __name__ == "__main__":
    main()