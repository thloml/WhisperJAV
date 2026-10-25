#!/usr/bin/env python3
"""
Persistent scene-transcription worker pool driver.

The community ctranslate2 ROCm build leaks host RAM per transcribe() call
inside long-running processes (~60MB/scene, unbounded). The only way to
reclaim that native memory is to kill the process. To stop the leak WITHOUT
reloading the ASR model on every batch (the model is ~3GB and must be loaded
again per process), we keep ONE persistent worker subprocess alive across many
batches and only recycle it (exit + respawn) after it has transcribed a
configurable number of scenes. This bounds leaked RAM to
max_scenes_per_worker * leak_rate while amortizing the model-load cost across
max_scenes_per_worker scenes instead of across just one batch of 50.

The worker talks to us over a simple stdin/stdout JSON control channel (see
scene_batch_worker.py). Each transcribe() call returns per-scene .srt files +
a <batch_id>.results.json on disk, exactly like the old one-shot worker.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Bounds leaked host RAM per worker while keeping the number of model reloads
# low. The worker is recycled when EITHER its peak RSS crosses
# max_rss_mb (computed as a percentage of the system's total RAM — the robust
# signal, it tracks the actual ct2 leak) or it has transcribed
# DEFAULT_MAX_SCENES_PER_WORKER scenes. Override with
# WHISPERJAV_MAX_SCENES_PER_WORKER / WHISPERJAV_MAX_WORKER_RSS_PCT.
DEFAULT_MAX_SCENES_PER_WORKER = int(
    os.environ.get("WHISPERJAV_MAX_SCENES_PER_WORKER", "200")
)
DEFAULT_MAX_WORKER_RSS_PCT = int(
    os.environ.get("WHISPERJAV_MAX_WORKER_RSS_PCT", "50")
)


def system_total_memory_mb() -> int:
    """Total physical RAM in MB, best-effort across platforms."""
    try:
        import psutil  # optional; prefer accurate cross-platform source
        return int(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages and pages > 0 and page_size and page_size > 0:
            return int(pages * page_size / (1024 * 1024))
    except Exception:
        pass
    return 32768  # conservative fallback: assume 32GB


class SceneBatchRunner:
    """Drives a persistent scene-batch worker subprocess with recycling."""

    def __init__(self, out_dir, task, cfg_path, max_scenes_per_worker=None,
                 max_worker_rss_pct=None):
        self.out_dir = Path(out_dir)
        self.task = task
        self.cfg_path = str(cfg_path)
        self.max_scenes = max_scenes_per_worker or DEFAULT_MAX_SCENES_PER_WORKER
        self.max_rss_pct = max_worker_rss_pct or DEFAULT_MAX_WORKER_RSS_PCT
        if not (1 <= self.max_rss_pct <= 100):
            self.max_rss_pct = DEFAULT_MAX_WORKER_RSS_PCT
        self.max_rss_mb = int(system_total_memory_mb() * self.max_rss_pct / 100)

        # The worker cfg JSON may carry "max_worker_rss_pct" (set by pipelines
        # from the CLI/GUI), so the limit survives without touching callers.
        if not max_worker_rss_pct:
            try:
                _cfg = json.loads(Path(self.cfg_path).read_text(encoding="utf-8"))
                if _cfg.get("max_worker_rss_pct"):
                    pct = int(_cfg["max_worker_rss_pct"])
                    if 1 <= pct <= 100:
                        self.max_rss_pct = pct
                        self.max_rss_mb = int(system_total_memory_mb() * pct / 100)
            except Exception:
                pass

        self._scenes_processed = 0
        self._proc = None
        # 1-based worker generation: bumped on every spawn. Generation N means
        # the model has been loaded N times; generation - 1 is how many times
        # the recogniser was replaced (scene budget, RSS ceiling, or a crash).
        self._generation = 0

    @property
    def worker_recycles(self) -> int:
        """Times the worker (and its loaded model) was replaced during this runner's life."""
        return max(0, self._generation - 1)
    def transcribe_batch(self, batch_path):
        """Send one batch to the worker and return its parsed results list.

        If the worker dies mid-batch (e.g. native crash), it is respawned and
        the batch retried once. The worker's model is loaded lazily in the
        subprocess, so the first batch pays the model-load cost.
        """
        for _attempt in range(2):
            if not self._alive():
                self._kill()
                self._spawn()
            try:
                self._write({"cmd": "transcribe", "batch": str(batch_path)})
            except Exception:
                self._kill()
                continue

            resp = self._read_done()
            if resp is not None:
                results = self._load_results(batch_path)
                # Stamp every scene with the worker generation that decoded it
                # (read before any recycle below, so it names the right worker).
                for _item in results:
                    if isinstance(_item, dict):
                        _item["worker_epoch"] = self._generation
                self._scenes_processed += resp.get("scenes", 0)
                rss_mb = int(resp.get("rss_mb", 0) or 0)
                if self._scenes_processed >= self.max_scenes or rss_mb >= self.max_rss_mb:
                    self._recycle()
                return results

            # Worker died or sent a non-done response: respawn and retry once.
            self._kill()
        return []

    def close(self):
        """Gracefully stop the worker (it exits via os._exit(0))."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._write({"cmd": "exit"})
                self._proc.wait(timeout=30)
            except Exception:
                self._kill()
        self._proc = None

    # ------------------------------------------------------------------ internals
    def _spawn(self):
        self._generation += 1
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "whisperjav.modules.scene_batch_worker",
             str(self.out_dir), self.task, self.cfg_path, str(self.max_scenes)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def _alive(self):
        return self._proc is not None and self._proc.poll() is None

    def _kill(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    def _write(self, obj):
        self._proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def _read_done(self):
        """Read control lines until a 'done' response. None on EOF/error.

        Stray non-JSON lines (defensive; the worker should keep its control
        channel clean) are skipped rather than treated as a failure.
        """
        while True:
            line = self._proc.stdout.readline()
            if not line:
                return None
            try:
                resp = json.loads(line)
            except Exception:
                continue  # skip stray non-JSON output
            if resp.get("cmd") == "done":
                return resp
            return None  # 'error' or unknown -> treat as failure

    def _load_results(self, batch_path):
        results_path = self.out_dir / f"{Path(batch_path).stem}.results.json"
        if not results_path.exists():
            return []
        try:
            return json.loads(results_path.read_text(encoding="utf-8")) or []
        except Exception:
            return []

    def _recycle(self):
        """Retire the current worker (frees leaked native memory) and respawn."""
        try:
            self._write({"cmd": "exit"})
            self._proc.wait(timeout=30)
        except Exception:
            self._kill()
        self._scenes_processed = 0
        self._proc = None
        self._spawn()
