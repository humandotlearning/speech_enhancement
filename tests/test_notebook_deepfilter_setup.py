import json
import unittest
from pathlib import Path


NOTEBOOK = Path("notebooks/audio_enhancement_experiments_colab.ipynb")
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


def notebook_code_cell_containing(path, needle):
    for cell in notebook_code_cells_for(path):
        if needle in cell:
            return cell
    raise AssertionError(f"Could not find code cell containing {needle!r} in {path}")


def notebook_text():
    return notebook_text_for(NOTEBOOK)


def notebook_code_cells():
    return notebook_code_cells_for(NOTEBOOK)


class DeepFilterNotebookSetupTest(unittest.TestCase):
    def test_deepfilternet_uses_release_binary_instead_of_python_package_install(self):
        text = notebook_text()

        self.assertNotIn("pip install -q deepfilternet", text)
        self.assertNotIn("pip install deepfilternet", text)
        self.assertIn("deep-filter-0.5.6-x86_64-unknown-linux-musl", text)
        self.assertIn("DEEP_FILTER_BINARY_URL", text)
        self.assertIn("install_deepfilter_binary", text)

    def test_deepfilternet_runner_uses_binary_output_dir_flag(self):
        code = "\n".join(notebook_code_cells())

        self.assertIn("DEEP_FILTER_BIN", code)
        self.assertIn('"--output-dir"', code)
        self.assertNotIn('"--out-dir"', code)

    def test_resemble_runner_falls_back_to_python_module_when_cli_is_not_on_path(self):
        code = "\n".join(notebook_code_cells())
        text = notebook_text()

        self.assertIn("git+https://github.com/resemble-ai/resemble-enhance.git", text)
        self.assertNotIn("%pip install -q resemble-enhance", text)
        self.assertIn("def get_resemble_command():", code)
        self.assertIn('"resemble-enhance"', code)
        self.assertIn('"resemble_enhance"', code)
        self.assertIn("shutil.which(executable_name)", code)
        self.assertIn("import importlib.util", code)
        self.assertIn('find_spec("resemble_enhance")', code)
        self.assertIn('"resemble_enhance.enhancer.__main__"', code)
        self.assertIn("cmd = get_resemble_command() + [in_dir, result_dir]", code)

    def test_comparison_view_includes_baseline_audio(self):
        code = "\n".join(notebook_code_cells())

        self.assertIn("Baseline audio from extracted video", code)
        self.assertIn('Audio(filename=str(BASELINE_WAV))', code)
        self.assertIn("if Path(BASELINE_WAV).exists():", code)


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

    def test_focused_notebook_upload_cell_matches_working_preview_pattern(self):
        upload_cell = notebook_code_cell_containing(REMUX_NOTEBOOK, "files.upload()")

        self.assertIn("if target_video.exists():", upload_cell)
        self.assertIn("target_video.unlink()", upload_cell)
        self.assertIn('print("Using video:", VIDEO_PATH)', upload_cell)
        self.assertIn('print("Video size MB:", round(VIDEO_PATH.stat().st_size / 1024 / 1024, 2))', upload_cell)
        self.assertNotIn("display(Video(", upload_cell)

    def test_focused_notebook_final_video_preview_uses_working_embed_pattern(self):
        preview_cell = notebook_code_cell_containing(REMUX_NOTEBOOK, "Final video with enhanced audio")

        self.assertIn("display(Video(str(FINAL_VIDEO), embed=False))", preview_cell)
        self.assertNotIn("display(Video(filename=str(FINAL_VIDEO), embed=True))", preview_cell)

    def test_focused_notebook_uses_resilient_model_setup(self):
        text = notebook_text_for(REMUX_NOTEBOOK)
        code = "\n".join(notebook_code_cells_for(REMUX_NOTEBOOK))

        self.assertNotIn("pip install -q deepfilternet", text)
        self.assertIn("deep-filter-0.5.6-x86_64-unknown-linux-musl", text)
        self.assertIn("git+https://github.com/resemble-ai/resemble-enhance.git", text)
        self.assertIn("def get_resemble_command():", code)
        self.assertIn('"resemble_enhance.enhancer.__main__"', code)


if __name__ == "__main__":
    unittest.main()
