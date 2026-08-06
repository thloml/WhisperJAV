#!/usr/bin/env python3
"""
Subprocess scene transcription worker.

The community ctranslate2 ROCm build leaks host RAM per transcribe() call
inside long-running processes (verified on gfx1201: ~60MB per scene, the
leak lives in the ct2 runtime and is NOT freed by model reloads). Running
scene batches in short-lived subprocesses bounds the leak: each worker's
native memory is reclaimed by the OS when it exits.

Invoked by the balanced/fidelity pipelines via:
    python -m whisperjav.modules.scene_batch_worker <batch.json> <out_dir> <task> <asr_cfg.json>

Writes one .srt per scene into out_dir and a <batch_id>.results.json with
per-scene status + VAD segments, then exits via os._exit(0) to skip
ctranslate2 destructors.
"""

import json
import os
import sys
from pathlib import Path


def main() -> int:
    batch_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    task = sys.argv[3]
    cfg = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))

    from whisperjav.modules.faster_whisper_pro_asr import FasterWhisperProASR
    from whisperjav.modules.whisper_pro_asr import WhisperProASR

    asr_class = cfg.get("asr_class", "faster_whisper")
    asr_cls = WhisperProASR if asr_class == "whisper_pro" else FasterWhisperProASR
    asr = asr_cls(
        model_config=cfg["model"],
        params=cfg["params"],
        task=task,
    )

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    results = []
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

    # Nuclear exit: skip ctranslate2 C++ destructors (crash risk), the OS
    # reclaims the leaked native memory.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
