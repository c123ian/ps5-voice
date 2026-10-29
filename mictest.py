import sounddevice as sd, numpy as np

sr = 16000
print(sd.query_devices(kind='input'))
input("Enter to record 3s...")
a = sd.rec(int(3*sr), samplerate=sr, channels=1, dtype='float32')
sd.wait()
print(a.shape, float(np.abs(a).max()))
