# Speech Enhancement Experiments

Google Colab notebook for testing speech enhancement pipelines on a source video. The notebook extracts the audio track, runs several enhancement branches, normalizes comparison outputs, and optionally remuxes enhanced audio back into MP4 files.

## What is included

- `notebooks/audio_enhancement_experiments_colab.ipynb` - the Colab notebook.
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

## Run in Google Colab

1. Open `notebooks/audio_enhancement_experiments_colab.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Run the dependency setup cells.
4. Upload the source video or mount Google Drive.
5. Run the six experiment cells.
6. Review the comparison table and playable audio/video widgets.
7. Download the generated ZIP archive from Colab.

The notebook writes its Colab outputs under `/content/audio_experiments/`.

## Local validation

From the repository root:

```powershell
python -m json.tool notebooks\audio_enhancement_experiments_colab.ipynb | Out-Null
python tests\test_notebook_deepfilter_setup.py
```

These checks validate notebook JSON structure and the expected setup for DeepFilterNet, Resemble Enhance, and baseline comparison playback.
