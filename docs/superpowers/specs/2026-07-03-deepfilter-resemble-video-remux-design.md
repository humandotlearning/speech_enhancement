# DeepFilter Resemble Video Remux Notebook Design

## Goal

Create a focused Google Colab notebook that takes one user-provided video, replaces its original audio with the best-performing enhancement chain from the experiment notebook, and produces a downloadable enhanced MP4.

## Context

The previous experiment notebook identified `DeepFilterNet -> Resemble Enhance` as the best output. The new notebook should reuse that proven process but remove the experiment branches that are no longer needed.

The notebook should follow the existing Colab input pattern:

- Option A: upload a video directly with the Colab file picker.
- Option B: mount Google Drive and copy a video from a configured Drive path.

## Notebook Shape

The deliverable is a new `.ipynb` notebook intended for Google Colab GPU runtime.

The notebook will be organized into these sections:

1. Install runtime dependencies.
2. Configure paths and settings.
3. Upload or mount the input video.
4. Define FFmpeg and filesystem helpers.
5. Install the DeepFilterNet release binary.
6. Extract baseline audio from the video.
7. Run DeepFilterNet, then Resemble Enhance.
8. Normalize the enhanced audio to a 48 kHz comparison WAV.
9. Remux the original video stream with the enhanced audio, replacing the original audio track.
10. Preview and download the final outputs.

## Processing Flow

The notebook will run this single workflow:

```text
input video
  -> FFmpeg extract mono 48 kHz WAV
  -> DeepFilterNet denoise
  -> Resemble Enhance
  -> FFmpeg convert comparison WAV
  -> FFmpeg remux original video stream + enhanced audio
  -> enhanced MP4
```

The video stream must be copied without re-encoding. The original audio stream must not be mapped into the final output. The enhanced audio should be encoded as AAC for MP4 compatibility.

## Input and Output Conventions

The notebook will use:

- `/content/deepfilter_resemble_video/input/` for uploaded or mounted input video.
- `/content/deepfilter_resemble_video/work/` for extracted and intermediate WAVs.
- `/content/deepfilter_resemble_video/outputs/` for final WAV and MP4 outputs.
- `/content/deepfilter_resemble_video/logs/` for command output.
- `/content/deepfilter_resemble_video/bin/` for the DeepFilterNet release binary.

The final video should be named `deepfilter_resemble_enhanced_video.mp4`.

## Dependency Strategy

Install dependencies directly in the Colab notebook. Use the same resilient DeepFilterNet strategy from the previous notebook: download the upstream `deep-filter` Linux release binary instead of installing the Python `deepfilternet` package.

Install Resemble Enhance from:

```text
git+https://github.com/resemble-ai/resemble-enhance.git
```

The Resemble runner should first look for the CLI names `resemble-enhance` and `resemble_enhance`, then fall back to `python -m resemble_enhance.enhancer.__main__`.

## Error Handling

The notebook should fail clearly when:

- no video is uploaded or the configured Drive path is missing,
- FFmpeg cannot extract audio,
- the DeepFilterNet binary cannot be downloaded or run,
- Resemble Enhance is not importable after installation,
- no WAV output is produced by either model,
- remuxing fails.

Command output should be saved to logs so Colab failures can be inspected after the fact.

## README Update

The README should describe both notebooks:

- the existing experiment notebook for comparing multiple branches,
- the new focused notebook for producing the chosen DeepFilterNet plus Resemble enhanced video.

The run instructions should explain that the focused notebook accepts a video by upload or Google Drive path and outputs an MP4 with the original video stream and replaced enhanced audio.

## Non-Goals

This notebook will not:

- run LavaSR or any other experiment branch,
- add objective audio metrics,
- build a reusable package,
- require local FFmpeg on the Windows workspace,
- re-encode the video stream unless FFmpeg cannot copy the original stream.

## Success Criteria

The setup is successful when the new notebook can:

1. Accept a video through upload or Google Drive path.
2. Extract the original audio to a 48 kHz WAV.
3. Run DeepFilterNet followed by Resemble Enhance.
4. Produce a 48 kHz comparison WAV.
5. Produce `deepfilter_resemble_enhanced_video.mp4` with the original video stream and enhanced replacement audio.
6. Show playable previews and create a ZIP archive for download.
