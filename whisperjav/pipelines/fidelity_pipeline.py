#!/usr/bin/env python3
"""V3 Architectur. Fidelity pipeline implementation - scene detection with WhisperPro ASR."""

import shutil
import sys 
from pathlib import Path
from typing import Dict, List
import time
from datetime import datetime

from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.whisper_pro_asr import WhisperProASR
from whisperjav.modules.srt_postprocessing import SRTPostProcessor as StandardPostProcessor

from whisperjav.modules.scene_detection_backends import SceneDetectorFactory

from whisperjav.modules.srt_stitching import SRTStitcher
from whisperjav.utils.logger import logger


from whisperjav.utils.progress_display import DummyProgress
from whisperjav.utils.progress_aggregator import AsyncProgressReporter
from whisperjav.utils.parameter_tracer import NullTracer

from whisperjav.modules.speech_enhancement import (
    create_enhancer_from_config,
    enhance_scenes,
    get_extraction_sample_rate,
    is_passthrough_backend,
)

class FidelityPipeline(BasePipeline):
    """Fidelity pipeline using scene detection with WhisperPro ASR (VAD-enhanced)."""
    
    def __init__(self, 
                 output_dir: str, 
                 temp_dir: str, 
                 keep_temp_files: bool, 
                 subs_language: str,
                 resolved_config: Dict,
                 progress_display=None,  
                 **kwargs):
        """
        Initializes the FidelityPipeline using V3 structured configuration.
        
        Args:
            output_dir: Output directory for subtitles
            temp_dir: Temporary directory for processing
            keep_temp_files: Whether to keep temporary files
            subs_language: Language for subtitles ('native' or 'direct-to-english')
            resolved_config: V3 structured configuration from TranscriptionTunerV3
            progress_display: Progress display object
            **kwargs: Additional parameters for base class
        """
        super().__init__(output_dir=output_dir, temp_dir=temp_dir, keep_temp_files=keep_temp_files, **kwargs)
        
        self.progress = progress_display or DummyProgress()
        self.subs_language = subs_language

        # Extract progress reporter and parameter tracer from kwargs
        self.progress_reporter = kwargs.get('progress_reporter', None)
        self.tracer = kwargs.get('parameter_tracer', NullTracer())

        # --- V3 STRUCTURED CONFIG UNPACKING ---
        model_cfg = resolved_config["model"]
        params = resolved_config["params"]
        features = resolved_config["features"]
        task = resolved_config["task"]
        
        # Set the ASR task based on the chosen output language
        self.asr_task = task  # Use the task from resolved config directly
        
        # Extract feature configurations
        scene_opts = features.get("scene_detection", {})
        post_proc_opts = features.get("post_processing", {})
        
        # Store params for metadata logging
        self.scene_detection_params = scene_opts
        '''
        self.scene_detection_params = {
            "detector_type": "AdaptiveSceneDetector",
            "using_defaults": True
        }        
        '''
        self.vad_params = params.get("vad", {})
        
        # Implement the smart model-switching logic (preserved from V2)
        effective_model_cfg = model_cfg.copy()
        if self.subs_language == 'direct-to-english' and model_cfg.get("model_name") == 'turbo':
            logger.info("Direct translation requested. Switching to 'large-v2' to perform translation.")
            effective_model_cfg["model_name"] = 'large-v2'
        # --- END V3 CONFIG UNPACKING ---

        # =================================================================
        # SCOPE-BASED RESOURCE MANAGEMENT (v1.7.3+)
        # GPU models are NOT created in __init__. We store CONFIGS only.
        # Models are created as LOCAL VARIABLES inside process() and
        # explicitly destroyed after use to prevent VRAM overlap.
        # =================================================================

        # Speech enhancement CONFIG (model created in process())
        self._enhancer_config = resolved_config  # Store full config for enhancer creation

        # Read enhancer backend to determine extraction sample rate
        enhancer_params = params.get("speech_enhancer", {})
        self._enhancer_backend_name = enhancer_params.get("backend", "none") or "none"
        self._enhancer_is_passthrough = is_passthrough_backend(self._enhancer_backend_name)
        self._enhance_for_vad = kwargs.get("enhance_for_vad", False)
        if self._enhance_for_vad and not self._enhancer_is_passthrough:
            logger.info("Enhance-for-VAD requested — enhancement will be applied to both VAD and ASR in fidelity pipeline")

        # v1.8.5+: Extract at 16kHz when enhancer is "none" (skip enhancement entirely)
        # Extract at 48kHz when a real enhancer is configured (enhancer needs high-SR)
        extraction_sr = get_extraction_sample_rate(self._enhancer_backend_name)
        self.audio_extractor = AudioExtractor(sample_rate=extraction_sr)
        self.scene_detector = SceneDetectorFactory.safe_create_from_legacy_kwargs(**scene_opts)

        # ASR CONFIG (model created in process() after enhancement cleanup)
        self._asr_config = {
            'model_config': effective_model_cfg,
            'params': params,
            'task': task,
            'tracer': self.tracer
        }
        # NOTE: self.asr is NOT created here - it's a local variable in process()

        self.stitcher = SRTStitcher()

        # Language code for post-processor and output filenames
        # Use 'en' for direct-to-english translation, otherwise use the selected source language
        if self.subs_language == 'direct-to-english':
            self.lang_code = 'en'
        else:
            # Get language from decoder params (set by CLI --language)
            self.lang_code = params["decoder"].get("language", "ja")
        self.standard_postprocessor = StandardPostProcessor(language=self.lang_code, **post_proc_opts)

    def process(self, media_info: Dict) -> Dict:
        """Process media file through fidelity pipeline with scene detection and VAD-enhanced ASR."""
        start_time = time.time()
        
        input_file = media_info['path']
        media_basename = media_info['basename']
        
        # Report file start if async reporter available
        if self.progress_reporter:
            self.progress_reporter.report_file_start(
                filename=media_basename,
                file_number=media_info.get('file_number', 1),
                total_files=media_info.get('total_files', 1)
            )
        
        master_metadata = self.metadata_manager.create_master_metadata(
            input_file=input_file,
            mode=self.get_mode_name(),
            media_info=media_info
        )

        # NOTE: reset_statistics moved to after ASR initialization (deferred loading)

        master_metadata["config"]["scene_detection_params"] = self.scene_detection_params
        master_metadata["config"]["vad_params"] = self.vad_params
        
        try:
            # Step 1: Extract audio
            if self.progress_reporter:
                self.progress_reporter.report_step("Transforming audio", 1, 6)
            self.progress.set_current_step("Transforming audio", 1, 6)
            
            audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
            extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)
            master_metadata["input_info"]["processed_audio_file"] = str(extracted_audio)
            master_metadata["input_info"]["audio_duration_seconds"] = duration
            self.metadata_manager.update_processing_stage(
                master_metadata, "audio_extraction", "completed", 
                output_path=str(audio_path), duration_seconds=duration)

            # Step 2: Detect scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Detecting audio scenes", 2, 6)
            self.progress.set_current_step("Detecting audio scenes", 2, 6)
            
            scenes_dir = self.temp_dir / "scenes"
            scenes_dir.mkdir(exist_ok=True)
            detection_result = self.scene_detector.detect_scenes(extracted_audio, scenes_dir, media_basename)
            scene_paths = detection_result.to_legacy_tuples()

            master_metadata["scenes_detected"] = []
            for idx, (scene_path, start_time_sec, end_time_sec, duration_sec) in enumerate(scene_paths):
                scene_info = {
                    "scene_index": idx, "filename": scene_path.name,
                    "start_time_seconds": round(start_time_sec, 3), 
                    "end_time_seconds": round(end_time_sec, 3),
                    "duration_seconds": round(duration_sec, 3), "path": str(scene_path)
                }
                master_metadata["scenes_detected"].append(scene_info)
            master_metadata["summary"]["total_scenes_detected"] = len(scene_paths)
            self.metadata_manager.update_processing_stage(
                master_metadata, "scene_detection", "completed",
                scene_count=len(scene_paths), scenes_dir=str(scenes_dir))

            # =================================================================
            # PHASE 1: SPEECH ENHANCEMENT (Exclusive VRAM Block)
            # When enhancer is "none" (passthrough), scenes are already at
            # 16kHz from extraction — skip this phase entirely.
            # When a real enhancer is configured, it runs as a LOCAL variable
            # created, used, and DESTROYED before ASR loads (VRAM Sandwich prevention).
            # =================================================================
            import gc

            self.progress.set_current_step("Preparing audio for ASR", 3, 6)

            if self._enhancer_is_passthrough:
                # v1.8.5+: Scenes already at 16kHz — skip enhancement entirely
                logger.info(
                    "Speech enhancer is passthrough — %d scenes at 16kHz, skipping enhancement",
                    len(scene_paths),
                )
                enhancer_name = "none"
                enhancer_display = "None (passthrough)"
            else:
                # NOTE: torch.cuda.is_available() is deliberately called ONLY
                # here. On ROCm builds initializing torch's HIP context before
                # the ctranslate2 model exists poisons ct2's host memory pool
                # (RAM leak per transcribe call — see get_ct2_gpu_device in
                # utils/device_detector.py). The ASR model loads AFTER this
                # block; the enhancer is torch-based, so the check belongs in
                # this branch. A torch-based enhancer followed by a ct2 ASR on
                # AMD still poisons ct2 (known limitation, see comment there).
                try:
                    import torch
                    _torch_available = torch.cuda.is_available()
                except ImportError:
                    _torch_available = False

                # A. Load Enhancer (always succeeds - "none" backend is fallback)
                enhancer = create_enhancer_from_config(self._enhancer_config)
                enhancer_name = enhancer.name
                enhancer_display = enhancer.display_name
                logger.info(f"Enhancing {len(scene_paths)} scenes with {enhancer_display}")

                def enhancement_progress(scene_num, total, name):
                    if scene_num == 1 or scene_num % 5 == 0 or scene_num == total:
                        pct = (scene_num / total) * 100
                        print(f"\rEnhancing: [{scene_num}/{total}] {pct:.0f}%", end='', flush=True)

                # B. Process Enhancement (includes 48kHz→16kHz resampling)
                scene_paths = enhance_scenes(
                    scene_paths,
                    enhancer,
                    self.temp_dir,
                    progress_callback=enhancement_progress,
                )
                print()  # Newline after progress

                # C. DESTROY Enhancer - This is the "JIT Unload"
                # We must confirm VRAM is near-zero before loading ASR
                logger.debug("Destroying enhancer to free VRAM before ASR load")
                enhancer.cleanup()
                del enhancer
                gc.collect()
                if _torch_available:
                    torch.cuda.empty_cache()
                    logger.debug("GPU memory cleared after enhancement - VRAM should be near-zero")

            master_metadata["config"]["speech_enhancement"] = {
                "enabled": not self._enhancer_is_passthrough,
                "backend": enhancer_name,
            }

            # =================================================================
            # PHASE 2: ASR TRANSCRIPTION (Subprocess Scene Batches)
            # Scenes are transcribed by short-lived worker subprocesses
            # (whisperjav.modules.scene_batch_worker). The community
            # ctranslate2 ROCm build leaks host RAM per transcribe() call in
            # long-lived processes (~60MB/scene, native) — a long video OOMs
            # the machine. Each worker process dies with its leaked memory,
            # bounding total RAM to BATCH_SCENES * leak_rate.
            # =================================================================

            logger.info("Initializing ASR model (subprocess workers)")

            # Step 4: Transcribe scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Transcribing scenes with VAD", 4, 6)
            self.progress.set_current_step("Transcribing scenes with VAD", 4, 6)

            scene_srts_dir = self.temp_dir / "scene_srts"
            scene_srts_dir.mkdir(exist_ok=True)
            scene_srt_info = []

            # Start scene transcription with unified progress management
            self.progress.start_subtask("Transcribing scenes", len(scene_paths))

            # Scene-level progress tracking variables
            last_update_time = time.time()
            transcription_start_time = time.time()
            update_interval = 30  # seconds
            total_scenes = len(scene_paths)
            BATCH_SCENES = 50

            # Print initial scene transcription header (always visible)
            print(f"\nTranscribing {total_scenes} scenes with VAD-enhanced processing:")

            logprob_filtered = 0
            nonverbal_filtered = 0

            import json as _json
            import subprocess as _subprocess

            # JSON-serializable ASR config for the workers
            asr_cfg = {
                "model": dict(self._asr_config["model_config"]),
                "params": self._asr_config["params"],
                "asr_class": "whisper_pro",
            }

            batches = [
                scene_paths[i:i + BATCH_SCENES]
                for i in range(0, total_scenes, BATCH_SCENES)
            ]
            done = 0
            for bi, batch in enumerate(batches):
                batch_items = [
                    (idx, sp, st)
                    for idx, (sp, st, _, _) in enumerate(scene_paths)
                    if any(sp == b[0] for b in batch)
                ]
                batch_map = {str(sp): (idx, sp, st) for idx, sp, st in batch_items}

                batch_path = scene_srts_dir / f"batch_{bi:04d}.json"
                cfg_path = scene_srts_dir / f"batch_{bi:04d}.cfg.json"
                results_path = scene_srts_dir / f"batch_{bi:04d}.results.json"
                batch_path.write_text(
                    _json.dumps([str(sp) for _, sp, _ in batch_items]), encoding="utf-8")
                cfg_path.write_text(_json.dumps(asr_cfg, default=str), encoding="utf-8")
                if results_path.exists():
                    results_path.unlink()

                try:
                    _subprocess.run(
                        [sys.executable, "-m", "whisperjav.modules.scene_batch_worker",
                         str(batch_path), str(scene_srts_dir), self.asr_task, str(cfg_path)],
                        timeout=3600,
                        check=False,
                    )
                except Exception as e:
                    logger.error(f"Scene batch {bi} failed to run: {e}")

                results = []
                if results_path.exists():
                    try:
                        results = _json.loads(results_path.read_text(encoding="utf-8"))
                    except Exception as e:
                        logger.error(f"Failed to read batch results {results_path}: {e}")

                for item in results:
                    scene = item.get("scene")
                    if scene not in batch_map:
                        continue
                    idx, scene_path, start_time_sec = batch_map[scene]
                    scene_num = idx + 1

                    fstats = item.get("filter_stats", {}) or {}
                    logprob_filtered += fstats.get('logprob_filtered', 0)
                    nonverbal_filtered += fstats.get('nonverbal_filtered', 0)

                    if item.get("ok"):
                        srt_path = Path(item["srt"])
                        if srt_path.exists() and srt_path.stat().st_size > 0:
                            scene_srt_info.append((srt_path, start_time_sec))
                            master_metadata["scenes_detected"][idx]["transcribed"] = True
                            master_metadata["scenes_detected"][idx]["srt_path"] = str(srt_path)
                        else:
                            master_metadata["scenes_detected"][idx]["transcribed"] = True
                            master_metadata["scenes_detected"][idx]["no_speech_detected"] = True
                    else:
                        master_metadata["scenes_detected"][idx]["transcribed"] = False
                        master_metadata["scenes_detected"][idx]["error"] = item.get("error", "unknown")
                        self.progress.show_message(
                            f"Scene {scene_num}/{total_scenes} failed: {item.get('error', '')}",
                            "error", 2.0)

                    done += 1
                    self.progress.update_subtask(1)

                # Mark any scenes the worker did not report as failed
                reported = {item.get("scene") for item in results}
                for idx, scene_path, _ in batch_items:
                    if str(scene_path) not in reported:
                        master_metadata["scenes_detected"][idx]["transcribed"] = False
                        master_metadata["scenes_detected"][idx]["error"] = "worker did not report scene"
                        self.progress.update_subtask(1)
                        done += 1

                # Show batch-level progress
                should_show_update = (
                    done == total_scenes or
                    bi == 0 or
                    time.time() - last_update_time > update_interval
                )
                if should_show_update:
                    progress_pct = (done / total_scenes) * 100
                    bar_width = 30
                    filled_width = int(bar_width * done / total_scenes)
                    progress_bar = '=' * filled_width + '-' * (bar_width - filled_width)
                    eta_text = ""
                    if done > 3:
                        elapsed = time.time() - transcription_start_time
                        avg_time_per_scene = elapsed / done
                        remaining_scenes = total_scenes - done
                        eta_seconds = remaining_scenes * avg_time_per_scene
                        if eta_seconds > 60:
                            eta_text = f" | ETA: {eta_seconds/60:.1f}m"
                        else:
                            eta_text = f" | ETA: {eta_seconds:.0f}s"
                    print(f"\rTranscribing: [{progress_bar}] {done}/{total_scenes} [{progress_pct:.1f}%]{eta_text}",
                          end='', flush=True)
                    last_update_time = time.time()

            self.progress.finish_subtask()

            # Print completion message for scene transcription (always visible)
            print(f"\n[DONE] Completed transcription of {total_scenes} scenes")
            
            # Step 5: Stitch scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Combining scene transcriptions", 5, 6)
            self.progress.set_current_step("Combining scene transcriptions", 5, 6)
            
            stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"
            num_subtitles = self.stitcher.stitch(scene_srt_info, stitched_srt_path)
            self.metadata_manager.update_processing_stage(
                master_metadata, "stitching", "completed", 
                subtitle_count=num_subtitles, output_path=str(stitched_srt_path))

            # Step 6: Post-process
            if self.progress_reporter:
                self.progress_reporter.report_step("Post-processing subtitles", 6, 6)
            self.progress.set_current_step("Post-processing subtitles", 6, 6)

            final_srt_path = self.output_dir / f"{media_basename}.{self.lang_code}.whisperjav.srt"
            processed_srt_path, stats = self.standard_postprocessor.process(stitched_srt_path, final_srt_path)
            
            # Ensure the final SRT is in the output directory
            if processed_srt_path != final_srt_path:
                shutil.copy2(processed_srt_path, final_srt_path)
                logger.debug(f"Copied final SRT from {processed_srt_path} to {final_srt_path}")
            
            # Move raw_subs folder to output directory
            temp_raw_subs_path = stitched_srt_path.parent / "raw_subs"
            if temp_raw_subs_path.exists():
                final_raw_subs_path = self.output_dir / "raw_subs"
                # Create raw_subs directory if it doesn't exist
                final_raw_subs_path.mkdir(exist_ok=True)
                
                # Copy only files related to current media_basename to avoid ghost files
                for file in temp_raw_subs_path.glob(f"{media_basename}*"):
                    dest_file = final_raw_subs_path / file.name
                    shutil.copy2(file, dest_file)
                    logger.debug(f"Copied {file.name} to raw_subs")
                
                logger.debug(f"Copied relevant raw_subs files to: {final_raw_subs_path}")

            self.metadata_manager.update_processing_stage(
                master_metadata, "postprocessing", "completed", statistics=stats, output_path=str(final_srt_path))

            master_metadata["output_files"]["final_srt"] = str(final_srt_path)
            master_metadata["output_files"]["stitched_srt"] = str(stitched_srt_path)
            master_metadata["summary"]["final_subtitles_refined"] = stats.get('total_subtitles', 0) - stats.get('empty_removed', 0)
            master_metadata["summary"]["final_subtitles_raw"] = num_subtitles
            master_metadata["summary"]["quality_metrics"] = {
                "hallucinations_removed": stats.get('removed_hallucinations', 0),
                "repetitions_removed": stats.get('removed_repetitions', 0),
                "duration_adjustments": stats.get('duration_adjustments', 0),
                "empty_removed": stats.get('empty_removed', 0)
            }

            # C. ASR CLEANUP: n/a — transcription ran in worker subprocesses
            # (scene_batch_worker) which exit via os._exit(0); their memory,
            # including the ctranslate2 ROCm host-RAM leak, is reclaimed by
            # the OS. Nothing to clean up in this process.
            logger.debug("ASR cleanup skipped (subprocess workers already exited)")

            master_metadata["summary"]["final_subtitles_raw"] += logprob_filtered + nonverbal_filtered
            master_metadata["summary"]["quality_metrics"].update({
                "logprob_filtered": logprob_filtered,
                "nonverbal_filtered": nonverbal_filtered,
                "cps_filtered": stats.get('cps_filtered', 0)
            })

            total_time = time.time() - start_time
            master_metadata["summary"]["total_processing_time_seconds"] = round(total_time, 2)
            master_metadata["metadata_master"]["updated_at"] = datetime.now().isoformat() + "Z"
            
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            self.cleanup_temp_files(media_basename)
            
            # Report completion
            if self.progress_reporter:
                self.progress_reporter.report_completion(
                    success=True,
                    stats={
                        'subtitles': master_metadata["summary"]["final_subtitles_refined"],
                        'duration': total_time,
                        'scenes': len(scene_paths)
                    }
                )
            
            return master_metadata
            
        except Exception as e:
            self.progress.show_message(f"Pipeline error: {str(e)}", "error", 0)
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.metadata_manager.update_processing_stage(
                master_metadata, "error", "failed", error_message=str(e))
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            
            # Report failure
            if self.progress_reporter:
                self.progress_reporter.report_completion(
                    success=False,
                    stats={'error': str(e)}
                )
            
            raise

    def get_mode_name(self) -> str:
        return "fidelity"

