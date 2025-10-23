import os
import re
import subprocess
import sys
import argparse
from pathlib import Path
from indextts.infer_v2 import IndexTTS2
import torch

# === PREVENT MPS MEMORY ERRORS ===
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# === CONFIGURATION ===
input_file = "飛蛾男孩.txt"          # Input text file
output_dir = "tts_output"            # Folder for generated audio
output_format = "mp3"                # "mp3" or "wav"
separator = "==="                    # High-level section separator
tempo_factor = 0.95                  # Slight slowdown (pitch preserved)
speaker_prompt = "examples/sample.wav"  # Voice reference audio
cfg_path = "checkpoints/config.yaml"
model_dir = "checkpoints"

# === PARSE ARGUMENTS ===
parser = argparse.ArgumentParser(description="Index-TTS2 batch synthesizer")
parser.add_argument(
    "segment",
    type=int,
    nargs="?",
    default=None,
    help="Specify a segment number to re-generate (1-based index). Leave empty to process all."
)
args = parser.parse_args()
regen_segment = args.segment

# === DEVICE AUTO-DETECT ===
if torch.backends.mps.is_available():
    device = "mps"
    device_name = "Apple Metal Performance Shaders"
elif torch.cuda.is_available():
    device = "cuda"
    device_name = "NVIDIA CUDA"
else:
    device = "cpu"
    device_name = "CPU"
print(f"💻 Using device: {device_name}")

# === INIT MODEL ===
print("🔧 Initializing IndexTTS2 model...")
tts = IndexTTS2(
    cfg_path=cfg_path,
    model_dir=model_dir,
    use_fp16=False,
    use_cuda_kernel=(device == "cuda"),
    use_deepspeed=False
)
tts.device = torch.device(device)
print("✅ Model loaded successfully!\n")

# === SETUP OUTPUT FOLDER ===
Path(output_dir).mkdir(exist_ok=True)

# === READ INPUT ===
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

sections = [seg.strip() for seg in text.split(separator) if seg.strip()]
total = len(sections)
print(f"📘 Found {total} major sections in '{input_file}'")

# === DETERMINE WHICH SECTIONS TO RUN ===
if regen_segment is not None:
    if regen_segment < 1 or regen_segment > total:
        sys.exit(f"❌ Invalid segment number: {regen_segment}. Must be between 1 and {total}.")
    sections_to_run = [regen_segment]
    print(f"🔁 Re-generating only segment #{regen_segment}")
else:
    sections_to_run = range(1, total + 1)
    print("🚀 Generating all sections")

# === PROCESS EACH SELECTED SECTION ===
for i in sections_to_run:
    section = sections[i - 1]
    prefix = re.sub(r"[^\w\u4e00-\u9fff]", "", section[:10]) or f"part{i}"
    base_name = f"{i:02d}_{prefix}"
    raw_path = Path(output_dir) / f"{base_name}_raw.{output_format}"
    slowed_path = Path(output_dir) / f"{base_name}.{output_format}"

    print(f"\n🎧 [{i}/{total}] Synthesizing → {base_name}")
    print("   Text preview:", section[:80].replace("\n", " "), "...")

    # 1️⃣ Generate speech
    tts.infer(
        speaker_prompt,
        text=section,
        output_path=str(raw_path),
        emo_vector=[0, 0, 0, 0, 0, 0, 0, 0],
        verbose=True
    )

    # 2️⃣ Apply tempo adjustment (pitch preserved)
    print(f"🎵  Applying tempo factor: {tempo_factor}x")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(raw_path),
        "-filter:a", f"atempo={tempo_factor}",
        str(slowed_path)
    ], check=True)

    os.remove(raw_path)
    print(f"✅ Saved slowed file: {slowed_path.name}")

print("\n🎉 Synthesis complete!")
print(f"🗂️  Output folder: {output_dir}")
