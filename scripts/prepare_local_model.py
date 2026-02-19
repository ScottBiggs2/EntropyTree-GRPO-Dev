"""
Prepare a local model directory with custom code that does not import dllm.
This avoids the circular import when loading the HuggingFace model with trust_remote_code.

Usage:
  python scripts/prepare_local_model.py [--out-dir ./model_cache]

Then set USE_LOCAL_MODEL_CODE=1 and LOCAL_MODEL_PATH=./model_cache (or your path)
when running validate_model.py, or pass the path to from_pretrained(local_path).
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_ID = "dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"
LOCAL_MODELING_FILE = ROOT / "model_custom_code" / "modeling_qwen2.py"


def main():
    parser = argparse.ArgumentParser(description="Download model and overwrite modeling file with dllm-free copy.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "model_cache",
        help="Directory to download the model into (default: ./model_cache)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()

    if not LOCAL_MODELING_FILE.exists():
        print(f"Error: {LOCAL_MODELING_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub required. pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {MODEL_ID} to {out_dir} ...")
    snapshot_download(MODEL_ID, local_dir=out_dir, local_dir_use_symlinks=False)

    dest = out_dir / "modeling_qwen2.py"
    shutil.copy2(LOCAL_MODELING_FILE, dest)
    print(f"Copied dllm-free modeling_qwen2.py to {dest}")

    print("Done. Load the model with:")
    print(f"  from_pretrained({out_dir!r}, trust_remote_code=True)")
    print("Or set env: USE_LOCAL_MODEL_CODE=1 LOCAL_MODEL_PATH=" + str(out_dir))


if __name__ == "__main__":
    main()
