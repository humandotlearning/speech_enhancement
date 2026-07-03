# DeepFilter Resemble Video Remux Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a focused Google Colab notebook that accepts a user video, runs DeepFilterNet followed by Resemble Enhance, and remuxes the original video stream with the enhanced replacement audio.

**Architecture:** The notebook is a single Colab artifact that reuses the proven model setup and runner patterns from the existing experiment notebook. It has one linear workflow and writes deterministic outputs, while a structural test verifies the notebook wiring without requiring Colab, FFmpeg, or model downloads locally.

**Tech Stack:** Google Colab, Python 3, FFmpeg, DeepFilterNet release binary, Resemble Enhance, `pandas`, `IPython.display`, `unittest`.

## Global Constraints

- Create a new `.ipynb` notebook intended for Google Colab GPU runtime.
- Keep the existing experiment notebook unchanged.
- Accept input video either by Colab upload or by Google Drive path.
- Use the DeepFilterNet release binary `deep-filter-0.5.6-x86_64-unknown-linux-musl`; do not install the Python `deepfilternet` package.
- Install Resemble Enhance from `git+https://github.com/resemble-ai/resemble-enhance.git`.
- Run exactly one default chain: `FFmpeg extract -> DeepFilterNet -> Resemble Enhance -> comparison WAV -> remux MP4`.
- Copy the original video stream during remux and map only the enhanced audio into the output MP4.
- Name the final video `deepfilter_resemble_enhanced_video.mp4`.
- Update the README to describe both the experiment notebook and the focused remux notebook.

---

## File Structure

- Create: `notebooks/deepfilter_resemble_video_remux_colab.ipynb`
  - Focused Colab notebook with setup, input, helpers, model runners, one processing workflow, preview, and archive cells.
- Modify: `tests/test_notebook_deepfilter_setup.py`
  - Add structural tests for the new focused notebook.
- Modify: `README.md`
  - Document the new notebook, its input options, and its final MP4 output.

---

### Task 1: Add Structural Tests for the Focused Notebook

**Files:**
- Modify: `tests/test_notebook_deepfilter_setup.py`

**Interfaces:**
- Consumes: `notebooks/deepfilter_resemble_video_remux_colab.ipynb`
- Produces: `DeepFilterResembleVideoRemuxNotebookTest`

- [ ] **Step 1: Write the failing tests**

Add a second notebook constant, helper, and test class:

```python
REMUX_NOTEBOOK = Path("notebooks/deepfilter_resemble_video_remux_colab.ipynb")


def notebook_text_for(path):
    return path.read_text(encoding="utf-8")


def notebook_code_cells_for(path):
    notebook = json.loads(notebook_text_for(path))
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
```

Use those helpers for the existing tests, then add:

```python
class DeepFilterResembleVideoRemuxNotebookTest(unittest.TestCase):
    def test_focused_notebook_exists_and_uses_winning_chain(self):
        text = notebook_text_for(REMUX_NOTEBOOK)
        code = "\n".join(notebook_code_cells_for(REMUX_NOTEBOOK))

        self.assertIn("DeepFilterNet -> Resemble Enhance", text)
        self.assertIn("run_deepfilternet(BASELINE_WAV", code)
        self.assertIn("run_resemble(deepfilter_wav", code)
        self.assertNotIn("run_lavasr", code)

    def test_focused_notebook_remux_replaces_original_audio(self):
        code = "\n".join(notebook_code_cells_for(REMUX_NOTEBOOK))

        self.assertIn("deepfilter_resemble_enhanced_video.mp4", code)
        self.assertIn('"-map", "0:v:0", "-map", "1:a:0"', code)
        self.assertIn('"-c:v", "copy"', code)
        self.assertIn('"-c:a", "aac"', code)
        self.assertNotIn('"-map", "0:a"', code)

    def test_focused_notebook_keeps_upload_or_drive_input_options(self):
        code = "\n".join(notebook_code_cells_for(REMUX_NOTEBOOK))

        self.assertIn("USE_GOOGLE_DRIVE = False", code)
        self.assertIn("DRIVE_VIDEO_PATH", code)
        self.assertIn("files.upload()", code)
        self.assertIn("drive.mount", code)

    def test_focused_notebook_uses_resilient_model_setup(self):
        text = notebook_text_for(REMUX_NOTEBOOK)
        code = "\n".join(notebook_code_cells_for(REMUX_NOTEBOOK))

        self.assertNotIn("pip install -q deepfilternet", text)
        self.assertIn("deep-filter-0.5.6-x86_64-unknown-linux-musl", text)
        self.assertIn("git+https://github.com/resemble-ai/resemble-enhance.git", text)
        self.assertIn("def get_resemble_command():", code)
        self.assertIn('"resemble_enhance.enhancer.__main__"', code)
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python tests\test_notebook_deepfilter_setup.py`

Expected: fail because `notebooks/deepfilter_resemble_video_remux_colab.ipynb` does not exist.

---

### Task 2: Create the Focused Notebook

**Files:**
- Create: `notebooks/deepfilter_resemble_video_remux_colab.ipynb`

**Interfaces:**
- Consumes: helper patterns from `notebooks/audio_enhancement_experiments_colab.ipynb`
- Produces: a valid Colab notebook containing the focused workflow

- [ ] **Step 1: Add notebook cells**

Create the notebook with these sections:

```markdown
# DeepFilterNet + Resemble Video Audio Replacement
```

Code cells must define:

```python
ROOT = Path("/content/deepfilter_resemble_video")
INPUT_DIR = ROOT / "input"
WORK_DIR = ROOT / "work"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"
BIN_DIR = ROOT / "bin"
FINAL_VIDEO = OUTPUT_DIR / "deepfilter_resemble_enhanced_video.mp4"
BASELINE_WAV = WORK_DIR / "baseline_48k.wav"
```

The workflow cell must run:

```python
deepfilter_wav = run_deepfilternet(BASELINE_WAV, WORK_DIR / "deepfilternet")
resemble_wav = run_resemble(deepfilter_wav, WORK_DIR / "resemble")
comparison_wav = make_comparison_wav(resemble_wav, OUTPUT_DIR / "deepfilter_resemble_comparison_48k.wav")
FINAL_VIDEO = remux_video(VIDEO_PATH, comparison_wav, OUTPUT_DIR / "deepfilter_resemble_enhanced_video.mp4")
```

- [ ] **Step 2: Preserve resilient helpers**

Copy and trim these helpers from the experiment notebook:

- `run_cmd`
- `safe_reset_dir`
- `existing_wavs`
- `newest_new_wav`
- `extract_audio`
- `make_comparison_wav`
- `remux_video`
- `install_deepfilter_binary`
- `run_deepfilternet`
- `get_resemble_command`
- `run_resemble`

The remux helper must include:

```python
[
    "ffmpeg", "-y", "-i", video_path, "-i", audio_wav,
    "-map", "0:v:0", "-map", "1:a:0",
    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
    "-shortest", output_mp4,
]
```

- [ ] **Step 3: Run tests and verify they pass**

Run: `python tests\test_notebook_deepfilter_setup.py`

Expected: all tests pass.

---

### Task 3: Update README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: new notebook path and output behavior
- Produces: README instructions for both notebooks

- [ ] **Step 1: Update README content**

Add `notebooks/deepfilter_resemble_video_remux_colab.ipynb` to "What is included".

Add a focused workflow section:

```markdown
## Focused DeepFilterNet + Resemble video workflow

Use `notebooks/deepfilter_resemble_video_remux_colab.ipynb` when you want the selected best chain only. It accepts a video by upload or Google Drive path, extracts the original audio, runs DeepFilterNet followed by Resemble Enhance, and writes `deepfilter_resemble_enhanced_video.mp4` with the original video stream and enhanced replacement audio.
```

Update local validation to include:

```powershell
python -m json.tool notebooks\deepfilter_resemble_video_remux_colab.ipynb | Out-Null
```

- [ ] **Step 2: Run validation**

Run:

```powershell
python -m json.tool notebooks\deepfilter_resemble_video_remux_colab.ipynb | Out-Null
python tests\test_notebook_deepfilter_setup.py
```

Expected: JSON validation exits 0 and all tests pass.
