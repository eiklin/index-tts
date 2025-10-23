from indextts.infer_v2 import IndexTTS2
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "我多年来一直不愿这么做。多年来，朋友和家人一直让我讲讲飞蛾男孩的故事，但我都拒绝了。或许他们以为我只是有一个令人毛骨悚然的轶事，或者一个有趣的知识点。没有多少人知道我从一开始就身在其中。"
tts.infer(spk_audio_prompt='examples/sample.wav', text=text, output_path="gen.wav", verbose=True)