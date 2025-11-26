from pathlib import Path
from watchdog.events import FileSystemEventHandler, FileSystemEvent, FileMovedEvent
from watchdog.observers.polling import PollingObserver as Observer
from queue import Queue, Empty
from threading import Thread

import subprocess
import time
import sys

_TASK_Q: "Queue[Path]" = Queue(maxsize=1024)

WATCHED_DIR = [
    Path(r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_sideview_smartfarm"),
    Path(r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_topview_smartfarm"),
]
EXTS = {".png", ".jpg", ".jpeg"}
SKIP_SUFFIXES = {".tmp", ".partial", ".crdownload"}

VIEW_MAP = {
    WATCHED_DIR[0].resolve(): "side",
    WATCHED_DIR[1].resolve(): "top",
}
    
# ป้องกันยิงซ้ำไฟล์เดิมในเวลาไล่เลี่ย
_LAST_RUN_TS: dict[str, float] = {}
MIN_GAP_SEC = 2.5

def infer_view_from_path(p: Path) -> str | None:
    # 1) แมตช์กับโฟลเดอร์ฐานที่กำหนดไว้
    pr = p.resolve()
    for base, v in VIEW_MAP.items():
        try:
            pr.relative_to(base)   # ถ้าไม่ raise แปลว่าอยู่ใต้ base
            return v
        except Exception:
            continue
    # 2) fallback heuristic
    s = str(pr).lower()
    parent_names = [pp.name.lower() for pp in list(pr.parents)[:4]]
    joined = " ".join([s] + parent_names)
    if any(k in joined for k in ["sideview", "side_view", "sideview_smartfarm"]) or "side" in parent_names:
        return "side"
    if any(k in joined for k in ["topview", "top_view", "topview_smartfarm"]) or "top" in parent_names:
        return "top"
    return None


def _worker():
    while True:
        p = _TASK_Q.get()
        try:
            _process_one(p)
        except Exception as e:
            log(f" - WORKER ERROR: {e}")
        finally:
            _TASK_Q.task_done()

def _process_one(p: Path):
    log(f"Event for: {p}")
    if should_skip(p):
        log("  - skip: suffix/ext not allowed")
        return
    if not p.exists():
        log("  - skip: path not exists (race)")
        return
    if debounce(str(p)):
        log("  - skip: debounced")
        return
    if not wait_until_quiescent_fast(p, min_quiet=1.0, max_quiet=2.0, timeout=5.0, poll=0.3):
        log("  - skip: not quiescent within timeout")
        return
    cmd = [sys.executable, "auto_run.py", "--input", str(p)]
    view = infer_view_from_path(p)
    if view:
        cmd += ["--view", view]
    project_root = Path(__file__).resolve().parent
    log(f"  - RUN: {' '.join(cmd)} (cwd={project_root})")
    try:
        res = subprocess.run(cmd, check=False, cwd=project_root)
        log(f"  - DONE: returncode={res.returncode}")
    except Exception as e:
        log(f"  - ERROR running pipeline: {e}")

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] {msg}", flush=True)

def wait_until_quiescent_fast(
    p: Path,
    min_quiet: float = 1.0,
    max_quiet: float = 3.0,
    timeout: float = 5.0,
    poll: float = 0.3
) -> bool:
    if not p.exists():
        return False
    
    import math
    start = time.time()
    last_sz = -1
    last_mt = -1.0
    stable_since = None
    last_t = time.time()
    
    while time.time() - start <= timeout:
        try:
            st = p.stat()
        except FileNotFoundError:
            return False
        
        sz = st.st_size
        mt = st.st_mtime
        now = time.time()
        dt = max(1e-6, now - last_t)
        growth_bps = 0 if last_sz < 0 else (sz - last_sz) / dt
        last_t = now
        
        target_quiet = max(min_quiet, min(max_quiet, min_quiet + (sz / 2_300_000 - 1.0)*0.4))
        if (sz != last_sz or abs(mt - last_mt) > 1e-6):
            stable_since = None
            last_sz, last_mt = sz, mt
        else:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= target_quiet:
                return True
        
        time.sleep(poll)
    
    return False

def should_skip(p: Path) -> bool:
    # ตัดไฟล์ที่ต่อท้ายด้วย suffix ชั่วคราว หรือสกุลไม่ตรง
    low = str(p).lower()
    if any(low.endswith(suf) for suf in SKIP_SUFFIXES):
        return True
    if p.suffix.lower() not in EXTS:
        return True
    return False

def debounce(path_str: str) -> bool:
    now = time.time()
    last = _LAST_RUN_TS.get(path_str, 0.0)
    if now - last < MIN_GAP_SEC:
        return True  # skip
    _LAST_RUN_TS[path_str] = now
    return False

def run_pipeline(p: Path):
    try:
        _TASK_Q.put_nowait(p)
    except Exception:
        log(f" - queue full: dropping task")
        
class Handler(FileSystemEventHandler):
    def on_created(self, event: FileSystemEvent):
        if event.is_directory:
            return
        log(f"on_created: {event.src_path}")
        try:
            run_pipeline(Path(event.src_path))
        except Exception as e:
            log(f"  - ERROR in on_created: {e}")

    def on_moved(self, event: FileMovedEvent):
        if event.is_directory:
            return
        log(f"on_moved: {event.src_path} -> {event.dest_path}")
        try:
            run_pipeline(Path(event.dest_path))
        except Exception as e:
            log(f"  - ERROR in on_moved: {e}")

    def on_modified(self, event: FileSystemEvent):
        if event.is_directory:
            return
        log(f"on_modified: {event.src_path}")
        try:
            run_pipeline(Path(event.src_path))
        except Exception as e:
            log(f"  - ERROR in on_modified: {e}")


if __name__ == "__main__":
    worker = Thread(target=_worker, daemon=True)
    worker.start()
    observer = Observer()
    for d in WATCHED_DIR:
        d.mkdir(parents=True, exist_ok=True)
        log(f"Watching directory: {d}")
        observer.schedule(Handler(), str(d), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()