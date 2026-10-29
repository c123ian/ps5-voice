import httpx

url = "http://192.168.0.97:8081/v1/chat/completions"
msgs = [{"role":"system","content":"You are a voice assistant. Reply in one short sentence."},
        {"role":"user","content":"hey"}]
body = dict(model="any", messages=msgs, chat_template_kwargs=dict(enable_thinking=False))
r = httpx.post(url, json=body, timeout=120)
r.raise_for_status()
print(r.json()["choices"][0]["message"]["content"])
