import base64, io, json, time, urllib.request
from PIL import Image, ImageDraw
img = Image.new("RGB", (448, 448), "white")
d = ImageDraw.Draw(img)
d.rectangle([40, 40, 200, 200], fill="red")
d.ellipse([250, 250, 410, 410], fill="blue")
buf = io.BytesIO(); img.save(buf, "PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
def chat(content, max_tokens):
    body = {"model": "thinkingmachines/Inkling-2-Mini", "temperature": 0,
            "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req))
    c = r["choices"][0]; u = r["usage"]
    print(f"--- {time.time()-t:.1f}s finish={c['finish_reason']} usage={u}")
    print(c["message"]["content"]); print()
chat("What is the capital of France? Answer in one sentence.", 400)
chat("Write a Python function that returns the n-th Fibonacci number iteratively.", 600)
chat([{"type": "image_url", "image_url": {"url": url}},
      {"type": "text", "text": "Describe the shapes in this image and their colors."}], 500)
