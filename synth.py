import os
import re
import subprocess
import argparse
from pathlib import Path
from indextts.infer_v2 import IndexTTS2
import torch

# === GPU MEMORY SAFETY FIX ===
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# === TORCH / CUDA OPTIMIZATION FLAGS ===
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# === CONFIGURATION ===
input_dir = Path("tts_input")
output_root = Path("tts_output")
output_format = "mp3"
separator = "==="
tempo_factor = 0.95
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
parser = argparse.ArgumentParser(description="Index-TTS2 batch synthesizer (A100 optimized, emotion control)")
parser.add_argument(
    "segment_positional",
    type=int,
    nargs="?",
    default=None,
    help="(Optional positional) Segment number to re-generate (1-based index)"
)
parser.add_argument(
    "--segment",
    type=int,
    help="(Optional flag) Segment number to re-generate (1-based index)"
)
parser.add_argument(
    "--emotion",
    type=str,
    choices=list(emotion_map.keys()),
    default=None,
    help="Optional: choose emotion (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm)"
)
args = parser.parse_args()

# Handle both positional and flag forms
regen_segment = args.segment if args.segment is not None else args.segment_positional
emo_vector = emotion_map[args.emotion] if args.emotion else [0.0] * 8

# === SUMMARY BANNER ===
print("\n" + "="*60)
print("🗂️  Index-TTS2 Batch Synthesis (A100 Optimized)")
print("="*60)
print(f"📂 Input folder : {input_dir.resolve()}")
print(f"💾 Output folder: {output_root.resolve()}")
print(f"🎭 Emotion      : {args.emotion or 'none (neutral)'}")
print(f"🎬 Segment      : {'all' if regen_segment is None else regen_segment}")
print(f"🎵 Tempo factor : {tempo_factor}")
print("="*60 + "\n")

# === INITIALIZE MODEL (A100 optimized) ===
print("🔧 Initializing IndexTTS2 model (FP16 + CUDA kernel)...")
tts = IndexTTS2(
    cfg_path=cfg_path,
    model_dir=model_dir,
    use_fp16=True,
    use_cuda_kernel=True,
    use_deepspeed=False
)
tts.device = torch.device("cuda")
print("✅ Model loaded successfully!\n")

# === WARM-UP INFERENCE ===
print("🔥 Performing warm-up inference (compiling CUDA kernels)...")
tts.infer(
    speaker_prompt,
    text="Warm-up run for CUDA optimization.",
    output_path="warmup.wav",
    emo_vector=emo_vector,
    verbose=False
)
print("✅ Warm-up complete. Proceeding with synthesis.\n")

# === FIND INPUT FILES ===
txt_files = sorted(input_dir.glob("*.txt"))
if not txt_files:
    raise SystemExit(f"❌ No .txt files found in {input_dir.resolve()}")
print(f"📘 Found {len(txt_files)} input file(s) in '{input_dir}'")

# === MAIN LOOP: process each input file ===
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
        section = sections[i - 1]
        prefix = re.sub(r"[^\w\u4e00-\u9fff]", "", section[:10]) or f"part{i}"
        base_name = f"{i:02d}_{prefix}"
        raw_path = output_dir / f"{base_name}_raw.{output_format}"
        slowed_path = output_dir / f"{base_name}.{output_format}"

        print(f"\n🎧 [{i}/{total}] Synthesizing → {story_name}/{base_name}")
        print("   Text preview:", section[:80].replace("\n", " "), "...")

        # 1️⃣ INFERENCE with selected (or neutral) emotion vector
        tts.infer(
            speaker_prompt,
            text=section,
            output_path=str(raw_path),
            emo_vector=emo_vector,
            verbose=True
        )

        # 2️⃣ APPLY TEMPO ADJUSTMENT (quiet FFmpeg)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-y",
            "-i", str(raw_path),
            "-filter:a", f"atempo={tempo_factor}",
            str(slowed_path)
        ], check=True)

        os.remove(raw_path)
        print(f"✅ Saved slowed file: {slowed_path.name}")

print("\n🎉 All files synthesized successfully!")
print(f"🗂️  Output root: {output_root.resolve()}")
