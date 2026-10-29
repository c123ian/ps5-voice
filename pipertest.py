import sounddevice as sd, numpy as np
from piper import PiperVoice, SynthesisConfig

v = PiperVoice.load("en_GB-vctk-medium.onnx")
cfg = SynthesisConfig(speaker_id=0)
chunks = [np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in v.synthesize("Hello, I am your PS5 assistant.", syn_config=cfg)]
audio = np.concatenate(chunks)
sd.play(audio, v.config.sample_rate); sd.wait()
