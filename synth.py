import os
import re
import subprocess
import argparse
from pathlib import Path
import torch

# === AUTO-MOUNT GOOGLE DRIVE IF IN COLAB ===
def mount_drive_if_needed():
    if os.path.exists("/content") and not os.path.exists("/content/drive/MyDrive"):
        try:
            import google.colab.drive as gdrive
            print("🔗 Mounting Google Drive...")
            gdrive.mount('/content/drive')
            print("✅ Google Drive mounted successfully.\n")
        except Exception as e:
            print(f"⚠️ Could not auto-mount Google Drive: {e}")
    else:
        print("📁 Google Drive already mounted or not running in Colab.\n")

mount_drive_if_needed()

from indextts.infer_v2 import IndexTTS2

# === GPU MEMORY SAFETY FIX ===
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# === TORCH / CUDA OPTIMIZATION FLAGS ===
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# === CONFIGURATION (Google Drive Paths) ===
input_dir = Path("/content/drive/MyDrive/index-tts/tts_input")
output_root = Path("/content/drive/MyDrive/index-tts/tts_output")
progress_log = output_root / "progress.log"
output_format = "mp3"
separator = "---"
tempo_factor = 0.9
speaker_prompt = "examples/sample.wav"
cfg_path = "checkpoints/config.yaml"
model_dir = "checkpoints"

# === DEFINE EMOTION VECTOR MAP ===
emotion_map = {
    "happy":       [1.0, 0, 0, 0, 0, 0, 0, 0],
    "angry":       [0, 1.0, 0, 0, 0, 0, 0, 0],
    "sad":         [0, 0, 1.0, 0, 0, 0, 0, 0],
    "afraid":      [0, 0, 0, 1.0, 0, 0, 0, 0],
    "disgusted":   [0, 0, 0, 0, 1.0, 0, 0, 0],
    "melancholic": [0, 0, 0, 0, 0, 1.0, 0, 0],
    "surprised":   [0, 0, 0, 0, 0, 0, 1.0, 0],
    "calm":        [0, 0, 0, 0, 0, 0, 0, 1.0],
}

# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser(description="Index-TTS2 batch synthesizer (A100 + Google Drive + checkpoint + auto-mount)")
parser.add_argument("segment_positional", type=int, nargs="?", default=None)
parser.add_argument("--segment", type=int, help="Segment number to re-generate (1-based index)")
parser.add_argument("--emotion", type=str, choices=list(emotion_map.keys()), default=None)
args = parser.parse_args()

regen_segment = args.segment if args.segment is not None else args.segment_positional
emo_vector = emotion_map[args.emotion] if args.emotion else [0.0] * 8

# === SUMMARY BANNER ===
print("\n" + "="*60)
print("🗂️  Index-TTS2 Batch Synthesis (A100 + Google Drive + Checkpoint)")
print("="*60)
print(f"📂 Input folder : {input_dir.resolve()}")
print(f"💾 Output folder: {output_root.resolve()}")
print(f"📘 Checkpoint   : {progress_log}")
print(f"🎭 Emotion      : {args.emotion or 'none (neutral)'}")
print(f"🎬 Segment      : {'all' if regen_segment is None else regen_segment}")
print(f"🎵 Tempo factor : {tempo_factor}")
print(f"🔹 Delimiter    : {separator}")
print("="*60 + "\n")

# === LOAD PROGRESS CHECKPOINT ===
completed_segments = set()
if progress_log.exists():
    with open(progress_log, "r", encoding="utf-8") as f:
        for line in f:
            completed_segments.add(line.strip())
    print(f"📑 Loaded checkpoint with {len(completed_segments)} completed segments.\n")
else:
    print("🆕 No checkpoint found. Starting fresh.\n")

# === INITIALIZE MODEL ===
print("🔧 Initializing IndexTTS2 model (FP16 + CUDA kernel)...")
tts = IndexTTS2(cfg_path=cfg_path, model_dir=model_dir, use_fp16=True, use_cuda_kernel=True, use_deepspeed=False)
tts.device = torch.device("cuda")
print("✅ Model loaded successfully!\n")

# === WARM-UP INFERENCE ===
print("🔥 Performing warm-up inference (compiling CUDA kernels)...")
tts.infer(speaker_prompt, text="Warm-up run for CUDA optimization.", output_path="warmup.wav", emo_vector=emo_vector, verbose=False)
print("✅ Warm-up complete. Proceeding with synthesis.\n")

# === FIND INPUT FILES ===
txt_files = sorted(input_dir.glob("*.txt"))
if not txt_files:
    raise SystemExit(f"❌ No .txt files found in {input_dir.resolve()}")
print(f"📘 Found {len(txt_files)} input file(s) in '{input_dir}'")

# === MAIN LOOP ===
for txt_path in txt_files:
    story_name = txt_path.stem
    output_dir = output_root / story_name
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n📘 Processing file: {txt_path.name}")
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    sections = [seg.strip() for seg in text.split(separator) if seg.strip()]
    total = len(sections)
    print(f"   ↳ Found {total} major sections")

    if regen_segment is not None:
        if regen_segment < 1 or regen_segment > total:
            raise SystemExit(f"❌ Invalid segment number: {regen_segment} (1–{total})")
        sections_to_run = [regen_segment]
        print(f"🔁 Re-generating only segment #{regen_segment}")
    else:
        sections_to_run = range(1, total + 1)

    for i in sections_to_run:
        checkpoint_key = f"{story_name}: segment {i}"
        if checkpoint_key in completed_segments and regen_segment is None:
            print(f"⏭️  Skipping {checkpoint_key} (already done)")
            continue

        section = sections[i - 1]
        prefix = re.sub(r"[^\w\u4e00-\u9fff]", "", section[:10]) or f"part{i}"
        base_name = f"{i:02d}_{prefix}"
        raw_path = output_dir / f"{base_name}_raw.{output_format}"
        slowed_path = output_dir / f"{base_name}.{output_format}"

        print(f"\n🎧 [{i}/{total}] Synthesizing → {story_name}/{base_name}")
        print("   Text preview:", section[:80].replace("\n", " "), "...")

        # 1️⃣ INFERENCE
        tts.infer(speaker_prompt, text=section, output_path=str(raw_path), emo_vector=emo_vector, verbose=True)

        # 2️⃣ APPLY TEMPO ADJUSTMENT (quiet FFmpeg)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-y", "-i", str(raw_path),
            "-filter:a", f"atempo={tempo_factor}",
            str(slowed_path)
        ], check=True)

        os.remove(raw_path)
        print(f"✅ Saved slowed file: {slowed_path.name}")

        # 3️⃣ UPDATE CHECKPOINT
        with open(progress_log, "a", encoding="utf-8") as logf:
            logf.write(f"{checkpoint_key}\n")
        completed_segments.add(checkpoint_key)
        print(f"📝 Updated checkpoint: {checkpoint_key}")

print("\n🎉 All files synthesized successfully!")
print(f"🗂️  Output root: {output_root.resolve()}")
print(f"🧾 Progress saved to: {progress_log}")
