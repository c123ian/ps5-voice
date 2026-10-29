import sounddevice as sd, numpy as np, httpx, threading, time
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig
from openwakeword.model import Model

sr = 16000
url = "http://192.168.0.97:8081/v1/chat/completions"
sys_msg = {"role":"system","content":"You are a voice assistant. Reply in one short sentence, written to be spoken aloud: conversational, with natural punctuation. An occasional 'hmm' or 'well,' is fine. No lists, markdown, or emoji."}

print("loading models...")
stt = WhisperModel("base", device="cpu", compute_type="int8")
voice = PiperVoice.load("en_GB-vctk-medium.onnx")
syn = SynthesisConfig(speaker_id=0)
ww = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

def record(chunk=1024, silence_sec=0.8, thresh=0.015, start_thresh=0.03, max_sec=30):
    "Record until ~silence_sec of quiet after speech, or max_sec."
    frames = []
    state = dict(started=False, silent=0.0, stop=False)
    def cb(indata, *a):
        frames.append(indata.copy())
        level = float(np.abs(indata).max())
        if level > start_thresh: state['started'] = True
        if state['started']: state['silent'] = state['silent'] + chunk/sr if level < thresh else 0.0
        if state['silent'] >= silence_sec: state['stop'] = True
    with sd.InputStream(samplerate=sr, channels=1, dtype='float32', blocksize=chunk, callback=cb):
        t0 = time.time()
        while not state['stop'] and time.time() - t0 < max_sec: time.sleep(0.05)
    return np.concatenate(frames)[:,0] if frames else np.zeros(1, dtype='float32')

def listen_for_wake(thresh=0.5, chunk=1280):
    "Listen passively until the wake word is heard."
    with sd.InputStream(samplerate=sr, channels=1, dtype='int16', blocksize=chunk) as stream:
        while True:
            data,_ = stream.read(chunk)
            pred = ww.predict(data[:,0])
            if pred.get("hey_jarvis", 0) > thresh: return

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
print("Listening for 'hey jarvis'... (Ctrl+C to quit)")
while True:
    listen_for_wake()
    print("  wake word! speak now...")
    a = record()
    text = transcribe(a)
    print(f"  you: {text}")
    if not text: continue
    reply = ask(text, hist)
    print(f"  ps5: {reply}")
    say(reply)
