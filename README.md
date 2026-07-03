# Speech Enhancement Experiments

Google Colab notebooks for testing and applying speech enhancement pipelines on a source video. The experiment notebook compares several enhancement branches; the focused notebook applies the selected DeepFilterNet plus Resemble Enhance workflow and remuxes the enhanced audio back onto the original video.

## What is included

- `notebooks/audio_enhancement_experiments_colab.ipynb` - the experiment Colab notebook for comparing multiple enhancement branches.
- `notebooks/deepfilter_resemble_video_remux_colab.ipynb` - the focused Colab notebook for producing an MP4 with DeepFilterNet plus Resemble enhanced replacement audio.
- `tests/test_notebook_deepfilter_setup.py` - structural tests for the notebook setup and expected model wiring.
- `docs/superpowers/specs/` - design notes for the notebook.
- `docs/superpowers/plans/` - implementation plan used to build the notebook.
- `examples/source/README.md` - source-video reference and expected local input location.

Generated archives and downloaded media binaries are intentionally ignored so the repository stays small and safe to publish.

## Source video

The example source video used during development came from:

https://www.youtube.com/watch?v=NHHIgaQJ4YM

Place a downloaded copy in `examples/source/` or upload it directly in Colab when the notebook asks for input. Media files in `examples/source/` are ignored by git.

## Experiments

The notebook runs these default branches:

1. DeepFilterNet -> Resemble Enhance
2. FFmpeg extract -> Resemble Enhance -> loudnorm -> remux
3. FFmpeg extract -> Resemble Enhance
4. DeepFilterNet or Resemble denoise -> LavaSR -> loudnorm
5. FFmpeg extract -> LavaSR
6. FFmpeg extract -> LavaSR -> loudnorm

Each successful branch creates a 48 kHz PCM comparison WAV. Branches that include remuxing also create an MP4 with the original video stream and enhanced audio.

## Focused DeepFilterNet + Resemble video workflow

Use `notebooks/deepfilter_resemble_video_remux_colab.ipynb` when you want the selected best chain only. It accepts a video by upload or Google Drive path, extracts the original audio, runs DeepFilterNet followed by Resemble Enhance, and writes `deepfilter_resemble_enhanced_video.mp4` with the original video stream and enhanced replacement audio.

The final MP4 maps the original video stream and the enhanced audio stream only, so the previous audio track is removed from the output.

## Run in Google Colab

1. Open `notebooks/audio_enhancement_experiments_colab.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Run the dependency setup cells.
4. Upload the source video or mount Google Drive.
5. Run the six experiment cells.
6. Review the comparison table and playable audio/video widgets.
7. Download the generated ZIP archive from Colab.

The notebook writes its Colab outputs under `/content/audio_experiments/`.

For the focused replacement workflow, open `notebooks/deepfilter_resemble_video_remux_colab.ipynb` instead. Use the upload cell for a local video, or set `USE_GOOGLE_DRIVE = True` and `DRIVE_VIDEO_PATH` for a Drive-hosted video. The notebook writes outputs under `/content/deepfilter_resemble_video/`.

## Local validation

From the repository root:

```powershell
python -m json.tool notebooks\audio_enhancement_experiments_colab.ipynb | Out-Null
python -m json.tool notebooks\deepfilter_resemble_video_remux_colab.ipynb | Out-Null
python tests\test_notebook_deepfilter_setup.py
```

These checks validate notebook JSON structure and the expected setup for DeepFilterNet, Resemble Enhance, and baseline comparison playback.
