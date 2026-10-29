import sounddevice as sd
from faster_whisper import WhisperModel

sr = 16000
model = WhisperModel("base", device="cpu", compute_type="int8")
input("Enter, then speak for 4s...")
a = sd.rec(int(4*sr), samplerate=sr, channels=1, dtype='float32')
sd.wait()
segs,_ = model.transcribe(a[:,0], language="en")
print(repr(" ".join(s.text for s in segs).strip()))
