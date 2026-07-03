import json
import unittest
from pathlib import Path


NOTEBOOK = Path("notebooks/audio_enhancement_experiments_colab.ipynb")


def notebook_text():
    return NOTEBOOK.read_text(encoding="utf-8")


def notebook_code_cells():
    notebook = json.loads(notebook_text())
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


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


if __name__ == "__main__":
    unittest.main()
