import sounddevice as sd, numpy as np, httpx, threading
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig

sr = 16000
url = "http://192.168.0.97:8081/v1/chat/completions"
sys_msg = {"role":"system","content":"You are a voice assistant. Reply in one short sentence."}

print("loading whisper...")
stt = WhisperModel("base", device="cpu", compute_type="int8")
voice = PiperVoice.load("en_GB-vctk-medium.onnx")
syn = SynthesisConfig(speaker_id=0)

def record():
    frames = []
    stop = threading.Event()
    def cb(indata, *a): frames.append(indata.copy())
    with sd.InputStream(samplerate=sr, channels=1, dtype='float32', callback=cb):
        input("  recording... Enter to stop")
        stop.set()
    return np.concatenate(frames)[:,0] if frames else np.zeros(1, dtype='float32')

def transcribe(a):
    segs,_ = stt.transcribe(a, language="en")
    return " ".join(s.text for s in segs).strip()

def ask(text, hist):
    hist.append({"role":"user","content":text})
    body = dict(model="any", messages=[sys_msg]+hist, chat_template_kwargs=dict(enable_thinking=False))
    r = httpx.post(url, json=body, timeout=120); r.raise_for_status()
    reply = r.json()["choices"][0]["message"]["content"]
    hist.append({"role":"assistant","content":reply})
    return reply

def say(text):
    chunks = [np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in voice.synthesize(text, syn_config=syn)]
    sd.play(np.concatenate(chunks), voice.config.sample_rate); sd.wait()

hist = []
while True:
    if input("\nEnter to talk (q to quit): ").strip().lower() == "q": break
    a = record()
    text = transcribe(a)
    print(f"  you: {text}")
    if not text: continue
    reply = ask(text, hist)
    print(f"  ps5: {reply}")
    say(reply)
