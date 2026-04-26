from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "kaggle_sync_config.json"
EXAMPLE_CONFIG_PATH = PROJECT_DIR / "kaggle_sync_config.example.json"
STAGE_DIR = PROJECT_DIR / ".kaggle_stage" / "kernel"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "kaggle_outputs"

SOURCE_FILES = [
    "prepare.py",
    "experiment.py",
    "run_project2.py",
    "program.md",
    "pyproject.toml",
]


def _run(cmd: list[str], cwd: Path | None = None, capture_output: bool = False):
    print("+", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        check=True,
        capture_output=capture_output,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def _run_with_retries(
    cmd: list[str],
    cwd: Path | None = None,
    capture_output: bool = False,
    retries: int = 3,
    sleep_seconds: int = 5,
):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return _run(cmd, cwd=cwd, capture_output=capture_output)
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            if attempt == retries:
                break
            print(f"Attempt {attempt}/{retries} failed: {exc}. Retrying in {sleep_seconds}s...")
            time.sleep(sleep_seconds)
    raise last_exc


def _kaggle_cmd() -> list[str]:
    explicit = os.environ.get("KAGGLE_EXE")
    if explicit:
        return [explicit]
    scripts_dir = Path(sys.executable).resolve().parent / "Scripts"
    kaggle_exe = scripts_dir / "kaggle.exe"
    if kaggle_exe.exists():
        return [str(kaggle_exe)]
    return ["kaggle"]


def _load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CONFIG_PATH.name}. Copy {EXAMPLE_CONFIG_PATH.name} to "
            f"{CONFIG_PATH.name} and fill in your Kaggle details."
        )
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    required = ["kernel_id", "kernel_title", "dataset_source", "dataset_mount_slug"]
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise RuntimeError(f"Missing required config keys: {', '.join(missing)}")
    return data


def _check_kaggle_cli():
    try:
        result = _run(_kaggle_cmd() + ["--version"], capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Kaggle CLI not found. Install with `pip install kaggle`."
        ) from exc
    print((result.stdout or result.stderr).strip())


def _dataset_mount_path(config):
    slug = config["dataset_mount_slug"].strip("/")
    subdir = config.get("dataset_subdir", "").strip("/")
    if subdir:
        return f"/kaggle/input/{slug}/{subdir}"
    return f"/kaggle/input/{slug}"


def _render_kernel_runner(config):
    dataset_root = _dataset_mount_path(config)
    prepare_src = (PROJECT_DIR / "prepare.py").read_text(encoding="utf-8")
    experiment_src = (PROJECT_DIR / "experiment.py").read_text(encoding="utf-8")
    run_src = (PROJECT_DIR / "run_project2.py").read_text(encoding="utf-8")

    prepare_literal = repr(prepare_src)
    experiment_literal = repr(experiment_src)
    run_literal = repr(run_src)

    eval_batch_size = int(config.get("eval_batch_size", 32))
    layers = config.get("layers", [])   # empty = all
    heads = config.get("heads", [])     # empty = all

    layers_args = (" --layers " + " ".join(str(l) for l in layers)) if layers else ""
    heads_args = (" --heads " + " ".join(str(h) for h in heads)) if heads else ""

    torch_index_url = config.get("p100_torch_index_url", "https://download.pytorch.org/whl/cu113")
    torch_version = config.get("p100_torch_version", "1.12.1+cu113")
    torchvision_version = config.get("p100_torchvision_version", "0.13.1+cu113")

    return f'''import os
import shlex
import sys
import tarfile
import zipfile
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

DEFAULT_DATASET_ROOT = r"{dataset_root}"
P100_TORCH_INDEX_URL = r"{torch_index_url}"
P100_TORCH_VERSION = r"{torch_version}"
P100_TORCHVISION_VERSION = r"{torchvision_version}"

PREPARE_SRC = {prepare_literal}
EXPERIMENT_SRC = {experiment_literal}
RUN_SRC = {run_literal}

Path("prepare.py").write_text(PREPARE_SRC, encoding="utf-8")
Path("experiment.py").write_text(EXPERIMENT_SRC, encoding="utf-8")
Path("run_project2.py").write_text(RUN_SRC, encoding="utf-8")
sys.path.insert(0, os.getcwd())


def _extract_if_needed(dataset_root: Path) -> Path:
    if (dataset_root / "splits").exists() and (dataset_root / "images").exists():
        return dataset_root
    extract_root = Path("/kaggle/working/indiwaste_dataset")
    extract_root.mkdir(parents=True, exist_ok=True)
    extracted_any = False
    for name in ("images", "splits", "annotations", "metadata", "docs"):
        zip_path = dataset_root / f"{{name}}.zip"
        tar_path = dataset_root / f"{{name}}.tar"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_root)
            extracted_any = True
        elif tar_path.exists():
            with tarfile.open(tar_path) as archive:
                archive.extractall(extract_root)
            extracted_any = True
    if extracted_any and (extract_root / "splits").exists() and (extract_root / "images").exists():
        return extract_root
    return dataset_root


os.environ.setdefault("INDIWASTE_ROOT", str(_extract_if_needed(Path(DEFAULT_DATASET_ROOT))))


def _gpu_name():
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except Exception as exc:
        print(f"gpu_probe_error: {{type(exc).__name__}}")
        return ""
    return (probe.stdout or probe.stderr or "").strip()


def _ensure_p100_compatible_torch():
    gpu_name = _gpu_name()
    if gpu_name:
        print(f"gpu_name: {{gpu_name}}")
    if "P100" not in gpu_name:
        return
    install_cmd = [
        sys.executable, "-m", "pip", "install",
        "--quiet", "--no-input", "--no-cache-dir", "--upgrade",
        "--index-url", P100_TORCH_INDEX_URL,
        f"torch=={{P100_TORCH_VERSION}}",
        f"torchvision=={{P100_TORCHVISION_VERSION}}",
    ]
    print("+", " ".join(install_cmd))
    try:
        subprocess.run(install_cmd, check=True)
        print("P100-compatible PyTorch installed successfully.")
    except Exception as exc:
        print(
            f"WARNING: Could not install P100-compatible PyTorch ({{exc.__class__.__name__}}: {{exc}}). "
            "Will attempt to run anyway -- training will fall back to CPU if CUDA is incompatible."
        )


_ensure_p100_compatible_torch()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


from experiment import main


if __name__ == "__main__":
    argv = [
        "run_project2.py",
        "--eval-batch-size", "{eval_batch_size}",
    ] + shlex.split("{layers_args}{heads_args}".strip())
    sys.argv = [a for a in argv if a]
    with open("run.log", "w", encoding="utf-8") as f:
        tee_out = Tee(sys.stdout, f)
        tee_err = Tee(sys.stderr, f)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            raise SystemExit(main())
'''


def _render_kernel_metadata(config):
    return {
        "id": config["kernel_id"],
        "title": config["kernel_title"],
        "code_file": "run_project2_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": str(bool(config.get("is_private", True))).lower(),
        "enable_gpu": str(bool(config.get("enable_gpu", True))).lower(),
        "enable_internet": str(bool(config.get("enable_internet", True))).lower(),
        "dataset_sources": [config["dataset_source"]],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def stage_kernel():
    config = _load_config()
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    for filename in SOURCE_FILES:
        src = PROJECT_DIR / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing required source file: {src}")
        shutil.copy2(src, STAGE_DIR / filename)

    runner_code = _render_kernel_runner(config)
    runner_path = STAGE_DIR / "run_project2_kaggle.py"
    runner_path.write_text(runner_code, encoding="utf-8")

    metadata_path = STAGE_DIR / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps(_render_kernel_metadata(config), indent=2),
        encoding="utf-8",
    )
    print(f"Staged Kaggle kernel at {STAGE_DIR}")
    print(f"Dataset mount path inside Kaggle: {_dataset_mount_path(config)}")


def push_kernel():
    _check_kaggle_cli()
    if not STAGE_DIR.exists():
        stage_kernel()
    _run(_kaggle_cmd() + ["kernels", "push", "-p", str(STAGE_DIR)])


def kernel_status():
    config = _load_config()
    _check_kaggle_cli()
    _run(_kaggle_cmd() + ["kernels", "status", config["kernel_id"]])


def download_output():
    config = _load_config()
    _check_kaggle_cli()
    output_dir = PROJECT_DIR / config.get("output_dir", "kaggle_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    _run_with_retries(
        _kaggle_cmd() + ["kernels", "output", config["kernel_id"], "-p", str(output_dir)]
    )
    print(f"Downloaded outputs to {output_dir}")


def watch_and_download():
    config = _load_config()
    _check_kaggle_cli()
    poll_seconds = int(config.get("poll_seconds", 30))
    kernel_id = config["kernel_id"]
    while True:
        result = _run_with_retries(
            _kaggle_cmd() + ["kernels", "status", kernel_id],
            capture_output=True,
        )
        text = (result.stdout or "") + (result.stderr or "")
        print(text.strip())
        lowered = text.lower()
        if "complete" in lowered:
            download_output()
            return
        if "error" in lowered or "failed" in lowered:
            raise RuntimeError("Kaggle kernel finished with an error state.")
        time.sleep(poll_seconds)


def prepare_config():
    if CONFIG_PATH.exists():
        print(f"{CONFIG_PATH.name} already exists.")
        return
    shutil.copy2(EXAMPLE_CONFIG_PATH, CONFIG_PATH)
    print(f"Created {CONFIG_PATH.name}. Fill it in before running push/status/output.")


def main():
    parser = argparse.ArgumentParser(description="Kaggle sync helper for Project 2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-config")
    subparsers.add_parser("stage-kernel")
    subparsers.add_parser("push")
    subparsers.add_parser("status")
    subparsers.add_parser("download-output")
    subparsers.add_parser("watch")

    args = parser.parse_args()
    if args.command == "prepare-config":
        prepare_config()
    elif args.command == "stage-kernel":
        stage_kernel()
    elif args.command == "push":
        push_kernel()
    elif args.command == "status":
        kernel_status()
    elif args.command == "download-output":
        download_output()
    elif args.command == "watch":
        watch_and_download()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
