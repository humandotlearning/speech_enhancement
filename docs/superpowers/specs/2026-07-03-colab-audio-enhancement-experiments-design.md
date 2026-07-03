# Colab Audio Enhancement Experiments Design

## Goal

Create a single Google Colab notebook for exploratory audio enhancement experiments on one input video. The notebook should run the requested enhancement chains, save intermediate WAV files, create remuxed MP4 outputs where requested, and make subjective A/B comparison easy.

## Context

The workspace currently contains one source video:

`vidssave.com 3 Things I’m Trying to Improve as a Mother 360P.mp4`

The local machine does not have `ffprobe` available, so video/audio inspection, extraction, loudness normalization, and remuxing should be handled inside Colab with FFmpeg.

The attached report recommends avoiding a blind stack of all models. The notebook should test the requested branches while preserving clear intermediate artifacts so outputs can be compared and the best route can be chosen manually.

## Notebook Shape

The deliverable is one `.ipynb` notebook intended for Google Colab GPU runtime.

The notebook will be organized into these sections:

1. Runtime and dependency setup
2. Upload or mount input video
3. FFmpeg audio extraction and baseline media inspection
4. Shared helper functions for shell commands, path management, loudnorm, and remux
5. Model setup for DeepFilterNet, Resemble Enhance, and LavaSR
6. Six experiment pipelines
7. Comparison table with playable audio/video outputs
8. Download/archive section
9. Optional rescue-model notes for VoiceFixer, VoiceRestore, and AudioSR

## Default Experiment Pipelines

The notebook will run these six pipelines:

1. DeepFilterNet -> Resemble Enhance
2. FFmpeg extract -> Resemble Enhance -> loudnorm -> remux
3. FFmpeg extract -> Resemble Enhance
4. DeepFilterNet or Resemble denoise -> LavaSR -> loudnorm
5. FFmpeg extract -> LavaSR
6. FFmpeg extract -> LavaSR -> loudnorm

Each pipeline will write outputs under `/content/audio_experiments/outputs/<pipeline_name>/`.

## Input and Output Conventions

The notebook will use:

- `/content/audio_experiments/input/` for uploaded or mounted input video
- `/content/audio_experiments/work/` for extracted and intermediate WAVs
- `/content/audio_experiments/outputs/` for final WAV/MP4 outputs
- `/content/audio_experiments/logs/` for command output and FFmpeg loudnorm logs

The baseline extracted WAV will be 48 kHz PCM. Speech models may internally resample, but each successful model output must also get a 48 kHz PCM comparison copy created with FFmpeg before it appears in the comparison table.

## Dependency Strategy

Install dependencies directly in Colab cells. Do not build a package or create reusable project scripts.

The default dependencies are:

- FFmpeg from the Colab environment or `apt`
- PyTorch from the Colab runtime
- `deepfilternet` for DeepFilterNet
- `resemble-enhance` for Resemble Enhance
- LavaSR installed from `git+https://github.com/ysharma3501/LavaSR.git`
- Python helper libraries: `soundfile`, `librosa`, `numpy`, `pandas`, `IPython`

The notebook should install model dependencies lazily enough that a failing optional model does not destroy already-produced outputs.

## Model Roles

DeepFilterNet is the denoise-first branch for noisy speech.

Resemble Enhance is the one-shot denoise/enhance branch and the second pass after DeepFilterNet in the first experiment.

LavaSR is the bandwidth restoration branch for dull, muffled, or low-bandwidth speech. It should be tested both alone and after denoise.

VoiceFixer, VoiceRestore, and AudioSR should be documented as optional rescue branches, not installed or executed by default, because the requested first experiment set only needs DeepFilterNet, Resemble Enhance, LavaSR, FFmpeg, loudnorm, and remuxing.

## Loudnorm and Remux

The notebook will include an FFmpeg loudnorm helper using a practical one-pass loudness normalization target:

- Integrated loudness: `I=-16`
- True peak: `TP=-1.5`
- Loudness range: `LRA=11`

For remuxed videos, FFmpeg will copy the video stream and replace the audio stream with the processed WAV/AAC audio. The original source video remains unchanged.

## Comparison Workflow

The notebook will build a comparison table with:

- Pipeline name
- Final WAV path
- Remuxed MP4 path where available
- Playable audio widget
- Notes field for manual listening observations

The user will choose the best output manually by listening. Objective scoring and speech transcription checks are intentionally out of scope for this first exploratory notebook.

## Error Handling

Each pipeline should run through a shared helper that records status, output paths, and error text. A failing branch should not stop subsequent independent branches from running.

The notebook should print clear next steps when:

- A runtime is CPU-only
- A model install fails
- A model output file is missing
- FFmpeg remux fails

## Non-Goals

This notebook will not:

- Build a reusable Python package
- Automate model choice
- Run all five listed restoration repos by default
- Train or fine-tune models
- Add objective audio metrics in the first version
- Require local FFmpeg on the Windows workspace

## Success Criteria

The setup is successful when the Colab notebook can:

1. Accept the input video through upload or Drive mount.
2. Extract a baseline 48 kHz WAV.
3. Run all six requested experiment branches, skipping only branches whose model installation fails.
4. Produce named WAV outputs for every successful branch.
5. Produce remuxed MP4 outputs for branches that include remux.
6. Show a compact comparison table for listening and downloading results.
