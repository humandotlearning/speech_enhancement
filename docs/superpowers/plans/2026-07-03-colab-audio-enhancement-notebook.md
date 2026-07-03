# Colab Audio Enhancement Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one Google Colab notebook that runs the six requested audio enhancement experiments on an uploaded video and saves comparison outputs.

**Architecture:** The notebook is a single exploratory artifact with setup cells, shared helper functions, model-specific runner functions, six pipeline cells, and a comparison/download section. Each pipeline writes to a stable output directory and records a status row so one failing branch does not block later branches.

**Tech Stack:** Google Colab, Python 3, FFmpeg, PyTorch, DeepFilterNet, Resemble Enhance, LavaSR, `soundfile`, `librosa`, `numpy`, `pandas`, `IPython.display`.

## Global Constraints

- Create one `.ipynb` notebook intended for Google Colab GPU runtime.
- Do not create a reusable Python package or repo-style script structure.
- Use FFmpeg inside Colab for inspection, extraction, loudnorm, comparison WAV conversion, and remuxing.
- Run exactly these default branches: `deepfilternet_resemble`, `resemble_loudnorm_remux`, `resemble_only`, `denoise_lavasr_loudnorm`, `lavasr_only`, `lavasr_loudnorm`.
- Save outputs under `/content/audio_experiments/outputs/<pipeline_name>/`.
- Save command logs under `/content/audio_experiments/logs/`.
- Create a 48 kHz PCM comparison WAV for every successful branch before adding it to the comparison table.
- Document VoiceFixer, VoiceRestore, and AudioSR as optional rescue branches, but do not install or run them by default.
- This workspace is not a git repository; skip commit steps and verify with file checks instead.

---

## File Structure

- Create: `notebooks/audio_enhancement_experiments_colab.ipynb`
  - Single Colab notebook deliverable.
  - Contains all setup, upload, extraction, model, pipeline, comparison, and archive cells.
- Existing: `docs/superpowers/specs/2026-07-03-colab-audio-enhancement-experiments-design.md`
  - Approved design source.
- Existing: `docs/superpowers/plans/2026-07-03-colab-audio-enhancement-notebook.md`
  - This implementation plan.

---

### Task 1: Create Notebook Artifact

**Files:**
- Create: `notebooks/audio_enhancement_experiments_colab.ipynb`

**Interfaces:**
- Consumes: approved design in `docs/superpowers/specs/2026-07-03-colab-audio-enhancement-experiments-design.md`
- Produces: notebook cells that define `run_cmd`, `ensure_dirs`, `extract_audio`, `make_comparison_wav`, `loudnorm_wav`, `remux_video`, `run_deepfilternet`, `run_resemble`, `run_lavasr`, `record_result`, and `show_results`

- [ ] **Step 1: Create the notebook directory**

Run: `New-Item -ItemType Directory -Force notebooks`

Expected: `notebooks` exists in the workspace.

- [ ] **Step 2: Add notebook metadata and markdown intro**

Create `notebooks/audio_enhancement_experiments_colab.ipynb` with Colab GPU metadata and an opening markdown cell:

```markdown
# Audio Enhancement Experiments for Video Speech

This Colab notebook runs six exploratory enhancement pipelines on one uploaded video:

1. DeepFilterNet -> Resemble Enhance
2. FFmpeg extract -> Resemble Enhance -> loudnorm -> remux
3. FFmpeg extract -> Resemble Enhance
4. DeepFilterNet or Resemble denoise -> LavaSR -> loudnorm
5. FFmpeg extract -> LavaSR
6. FFmpeg extract -> LavaSR -> loudnorm
```

Expected: opening cell names the six default experiments exactly.

- [ ] **Step 3: Add runtime setup cells**

Add cells that:

```python
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from IPython.display import Audio, Video, display, HTML
```

and:

```python
ROOT = Path("/content/audio_experiments")
INPUT_DIR = ROOT / "input"
WORK_DIR = ROOT / "work"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"

for directory in [INPUT_DIR, WORK_DIR, OUTPUT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RESULTS = []
VIDEO_PATH = None
BASELINE_WAV = WORK_DIR / "baseline_48k.wav"
```

Expected: paths match the spec.

- [ ] **Step 4: Add shared helper cells**

Add code cells defining:

```python
def run_cmd(cmd, log_name=None, check=True):
    print("+", " ".join(str(part) for part in cmd))
    proc = subprocess.run(
        [str(part) for part in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if log_name:
        (LOG_DIR / log_name).write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0 and check:
        raise RuntimeError(proc.stdout)
    return proc.stdout
```

```python
def extract_audio(video_path, output_wav=BASELINE_WAV):
    run_cmd([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "48000", "-sample_fmt", "s16",
        output_wav,
    ], "extract_audio.log")
    return Path(output_wav)
```

```python
def make_comparison_wav(input_wav, output_wav):
    run_cmd([
        "ffmpeg", "-y", "-i", input_wav,
        "-ac", "1", "-ar", "48000", "-sample_fmt", "s16",
        output_wav,
    ], f"{Path(output_wav).stem}_comparison.log")
    return Path(output_wav)
```

```python
def loudnorm_wav(input_wav, output_wav, i=-16, tp=-1.5, lra=11):
    run_cmd([
        "ffmpeg", "-y", "-i", input_wav,
        "-af", f"loudnorm=I={i}:TP={tp}:LRA={lra}",
        "-ac", "1", "-ar", "48000", "-sample_fmt", "s16",
        output_wav,
    ], f"{Path(output_wav).stem}_loudnorm.log")
    return Path(output_wav)
```

```python
def remux_video(video_path, audio_wav, output_mp4):
    run_cmd([
        "ffmpeg", "-y", "-i", video_path, "-i", audio_wav,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_mp4,
    ], f"{Path(output_mp4).stem}_remux.log")
    return Path(output_mp4)
```

Expected: each helper writes logs and returns a `Path`.

- [ ] **Step 5: Add model setup and runner cells**

Add install cells for:

```python
!apt-get update -qq
!apt-get install -y -qq ffmpeg
!pip install -q --upgrade pip
!pip install -q deepfilternet resemble-enhance soundfile librosa pandas
!pip install -q git+https://github.com/ysharma3501/LavaSR.git
```

Add runners for DeepFilterNet, Resemble Enhance, and LavaSR. DeepFilterNet and Resemble runners should call their CLIs through `run_cmd`; LavaSR should use `LavaEnhance2` from Python and write 48 kHz WAV.

Expected: model functions accept an input WAV and output directory, and return the produced WAV path.

- [ ] **Step 6: Add six pipeline cells**

Add one cell per default pipeline with stable output names:

```python
run_pipeline("deepfilternet_resemble", pipeline_deepfilternet_resemble)
run_pipeline("resemble_loudnorm_remux", pipeline_resemble_loudnorm_remux)
run_pipeline("resemble_only", pipeline_resemble_only)
run_pipeline("denoise_lavasr_loudnorm", pipeline_denoise_lavasr_loudnorm)
run_pipeline("lavasr_only", pipeline_lavasr_only)
run_pipeline("lavasr_loudnorm", pipeline_lavasr_loudnorm)
```

Expected: each pipeline records `pipeline`, `status`, `final_wav`, `comparison_wav`, `remuxed_mp4`, and `error`.

- [ ] **Step 7: Add comparison and archive cells**

Add cells that display a DataFrame, audio widgets for successful comparison WAVs, video widgets for successful MP4s, and create a ZIP archive:

```python
results_df = pd.DataFrame(RESULTS)
display(results_df)
```

```python
archive_path = shutil.make_archive(str(ROOT / "audio_experiment_outputs"), "zip", ROOT)
print(archive_path)
```

Expected: successful outputs are directly playable in Colab and downloadable as a ZIP.

- [ ] **Step 8: Add optional rescue notes**

Add markdown explaining that VoiceFixer, VoiceRestore, and AudioSR are excluded from default runs and can be added later for badly degraded, clipped, reverberant, or non-speech-heavy material.

Expected: notes do not install or execute those models.

---

### Task 2: Validate Notebook Structure

**Files:**
- Test: `notebooks/audio_enhancement_experiments_colab.ipynb`

**Interfaces:**
- Consumes: notebook from Task 1
- Produces: validation evidence that the notebook is valid JSON, has Colab metadata, contains required pipelines, and avoids default VoiceFixer/VoiceRestore/AudioSR installs

- [ ] **Step 1: Validate JSON can load**

Run:

```powershell
python -m json.tool notebooks\audio_enhancement_experiments_colab.ipynb > $null
```

Expected: command exits with code 0.

- [ ] **Step 2: Validate required strings**

Run:

```powershell
rg -n "deepfilternet_resemble|resemble_loudnorm_remux|resemble_only|denoise_lavasr_loudnorm|lavasr_only|lavasr_loudnorm" notebooks\audio_enhancement_experiments_colab.ipynb
```

Expected: all six pipeline names appear.

- [ ] **Step 3: Validate optional rescue models are notes only**

Run:

```powershell
rg -n "pip install.*voicefixer|pip install.*voicerestore|pip install.*audiosr|git\\+https://github.com/haoheliu/voicefixer|git\\+https://github.com/skirdey/voicerestore|git\\+https://github.com/haoheliu/versatile_audio_super_resolution" notebooks\audio_enhancement_experiments_colab.ipynb
```

Expected: no matches.

- [ ] **Step 4: Confirm workspace is not a git repository**

Run:

```powershell
git status --short
```

Expected: `fatal: not a git repository (or any of the parent directories): .git`.

No commit is made in this workspace.
