#!/usr/bin/env python3
"""
Persistent subprocess scene transcription worker.

The community ctranslate2 ROCm build leaks host RAM per transcribe() call
inside long-running processes (verified on gfx1201: ~60MB per scene, the leak
lives in the ct2 runtime and is NOT freed by model reloads). To benefit from
the subprocess workaround WITHOUT paying the model-load cost on every batch,
this worker is PERSISTENT: it loads the model once and then handles many
batches over a simple stdin/stdout JSON control channel. The parent
(SceneBatchRunner) tells it to exit once a configurable number of scenes has
been transcribed, and the OS reclaims the leaked native memory.

Invoked by the balanced/fidelity pipelines via SceneBatchRunner:
    python -m whisperjav.modules.scene_batch_worker <out_dir> <task> <cfg.json> [max_scenes]

Control protocol (stdin -> stdout, one JSON object per line):
  in:  {"cmd": "transcribe", "batch": "<batch.json>"}
  out: {"cmd": "done", "batch": "<batch.json>", "scenes": N, "ok": M}
  in:  {"cmd": "exit"}

Each transcribe writes one .srt per scene into out_dir and a
<batch_id>.results.json with per-scene status + VAD segments, then exits via
os._exit(0) to skip ctranslate2 destructors.
"""

import json
import os
import resource
import sys
from pathlib import Path


def _open_control():
    # Hold a handle to the REAL stdout (the control channel) before redirecting
    # sys.stdout, so library prints (faster-whisper/ctranslate2) cannot corrupt
    # the protocol. Stray prints go to stderr instead.
    return os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)


def _send(control, obj):
    control.write(json.dumps(obj, ensure_ascii=False) + "\n")
    control.flush()


def main() -> int:
    # Silence the harmless urllib3/chardet version-mismatch warning that
    # requests emits on every import; it floods the parent's terminal from the
    # worker's inherited stderr and is unrelated to transcription.
    import warnings as _warnings
    _warnings.filterwarnings("ignore", message=r".*doesn't match a supported version.*")

    out_dir = Path(sys.argv[1])
    task = sys.argv[2]
    cfg = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))

    # The control channel is the REAL stdout (fd 1). Grab it BEFORE redirecting
    # sys.stdout below.
    control = _open_control()

    # CRITICAL: redirect sys.stdout -> sys.stderr BEFORE importing whisperjav.
    # The logger binds logging.StreamHandler(sys.stdout) at import time, so if
    # we redirect too late, every whisperjav log line (e.g. "Speech Segmenter
    # initialized") would be written into the control channel and corrupt the
    # JSON protocol — the parent would then treat every batch as failed and
    # produce empty subtitles.
    sys.stdout = sys.stderr

    from whisperjav.modules.faster_whisper_pro_asr import FasterWhisperProASR
    from whisperjav.modules.whisper_pro_asr import WhisperProASR

    asr_class = cfg.get("asr_class", "faster_whisper")
    asr_cls = WhisperProASR if asr_class == "whisper_pro" else FasterWhisperProASR
    asr = asr_cls(
        model_config=cfg["model"],
        params=cfg["params"],
        task=task,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except Exception:
            _send(control, {"cmd": "error", "error": "invalid command"})
            continue

        if cmd.get("cmd") == "exit":
            break
        if cmd.get("cmd") != "transcribe":
            _send(control, {"cmd": "error", "error": "unknown command"})
            continue

        batch_path = Path(cmd["batch"])
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        results = []
        ok_count = 0
        for scene in batch:
            scene_path = Path(scene)
            srt_path = out_dir / f"{scene_path.stem}.srt"
            try:
                asr.transcribe_to_srt(scene_path, srt_path, task=task)
                results.append({
                    "scene": scene,
                    "ok": True,
                    "srt": str(srt_path),
                    "vad": getattr(asr, "get_last_vad_segments", lambda: [])() or [],
                    "filter_stats": asr.get_filter_statistics(),
                })
                ok_count += 1
            except Exception as e:
                results.append({
                    "scene": scene,
                    "ok": False,
                    "error": str(e),
                    "vad": [],
                    "filter_stats": {},
                })

        results_path = out_dir / f"{batch_path.stem}.results.json"
        results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

        _send(control, {
            "cmd": "done",
            "batch": str(batch_path),
            "scenes": len(batch),
            "ok": ok_count,
            # Peak resident set (MB) — monotonic, so it tracks the accumulated
            # ctranslate2 leak. The parent uses this to recycle the worker when
            # real memory (not just a fixed scene count) crosses a threshold.
            "rss_mb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        })

    # Nuclear exit: skip ctranslate2 C++ destructors (crash risk), the OS
    # reclaims the leaked native memory.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
