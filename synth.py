import os
import subprocess
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

# === DEVICE AUTO-DETECT ===
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"💻 Using device: {device}")

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

# Split only by major logical sections
sections = [seg.strip() for seg in text.split(separator) if seg.strip()]
print(f"📘 Found {len(sections)} major sections in '{input_file}'")

# === PROCESS EACH SECTION ===
for i, section in enumerate(sections, start=1):
    raw_path = Path(output_dir) / f"{i:02d}_raw.{output_format}"
    slowed_path = Path(output_dir) / f"{i:02d}.{output_format}"

    if slowed_path.exists():
        print(f"⏭️  Skipping [{i}] already exists: {slowed_path.name}")
        continue

    print(f"\n🎧 [{i}/{len(sections)}] Synthesizing section ({len(section)} chars)...")
    print("   Text preview:", section[:80].replace("\n", " "), "...")

    # 1️⃣ Generate speech (IndexTTS2 handles internal segmentation)
    tts.infer(
        speaker_prompt,         # positional argument
        text=section,
        output_path=str(raw_path),
        verbose=True
    )

    # 2️⃣ Apply tempo adjustment (slightly slower, pitch preserved)
    print(f"🎵  Applying tempo factor: {tempo_factor}x")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(raw_path),
        "-filter:a", f"atempo={tempo_factor}",
        str(slowed_path)
    ], check=True)

    # 3️⃣ Clean up temporary file
    os.remove(raw_path)
    print(f"✅ Saved slowed file: {slowed_path.name}")

print("\n🎉 All sections synthesized successfully!")
print(f"🗂️  Output folder: {output_dir}")
