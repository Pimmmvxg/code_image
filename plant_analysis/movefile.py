# movefile.py
from __future__ import annotations
from pathlib import Path
import threading, queue, shutil, time, os, atexit
from typing import Optional, Tuple

class BackgroundCopier:
    def __init__(self, pending_dir: str = "C:/results_pending_to_R", max_workers: int = 1):
        self.q: "queue.Queue[Tuple[str,str,int,float]]" = queue.Queue()
        self.pending_dir = Path(pending_dir)
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.workers = []
        self._stop = threading.Event()
        for _ in range(max_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)

    @staticmethod
    def _atomic_copy(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            try: dst.unlink()
            except Exception: pass
        shutil.copy2(src, dst)
        if dst.stat().st_size != src.stat().st_size:
            try: dst.unlink()
            except Exception: pass
            raise IOError(f"Size mismatch after direct copy: {src} -> {dst}")

    def _worker(self):
        while not self._stop.is_set():
            try:
                src, dst, retries, backoff = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                src_p, dst_p = Path(src), Path(dst)
                ok = False; last_err: Optional[Exception] = None
                for i in range(retries):
                    try:
                        self._atomic_copy(src_p, dst_p)
                        ok = True; break
                    except Exception as e:
                        last_err = e
                        time.sleep(backoff * (2 ** i))
                if not ok:
                    pend = self.pending_dir / dst_p.name
                    try:
                        shutil.copy2(src_p, pend)
                        print(f"[Move to R:] Failed -> kept pending: {pend}; err={last_err}")
                    except Exception as ee:
                        print(f"[Move to R:] Also failed to keep pending: {pend}; err={ee}")
                else:
                    print(f"[Move to R:] Copied: {dst_p}")
            finally:
                self.q.task_done()

    def enqueue(self, src: str, dst: str, retries: int = 3, backoff: float = 1.6):
        self.q.put((src, dst, retries, backoff))

    def shutdown(self, wait: bool = False):
        self._stop.set()
        if wait:
            for t in self.workers:
                t.join(timeout=1.0)

# สร้าง singleton ไว้ใช้ทั้งโปรเจ็กต์
_bg_copier = BackgroundCopier(pending_dir="C:/results_pending_to_R", max_workers=1)

def enqueue_copy(src: str, dst: str, retries: int = 3, backoff: float = 1.6):
    _bg_copier.enqueue(src, dst, retries=retries, backoff=backoff)

# ระบายคิวก่อนจบprocess (กันงานค้าง)
def _drain_on_exit():
    try: _bg_copier.q.join()
    except Exception: pass
atexit.register(_drain_on_exit)
