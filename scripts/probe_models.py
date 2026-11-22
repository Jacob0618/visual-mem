"""Probe: gemini-3.8-flash (text+image, thinking config, usage) and gemini-embedding-2 (text, image, cross-modal)."""
import sys, io, json, time
sys.path.insert(0, "src")
from keys import gemini_key
from google import genai
from google.genai import types
from PIL import Image, ImageDraw
import numpy as np

c = genai.Client(api_key=gemini_key())

def mkimg(text, color):
    im = Image.new("RGB", (512, 384), color)
    d = ImageDraw.Draw(im)
    d.rectangle([40, 120, 470, 260], fill="white")
    d.text((60, 170), text, fill="black")
    b = io.BytesIO(); im.save(b, format="PNG"); return b.getvalue()

img_a = mkimg("PARKING LEVEL B, SPOT 42", (200, 30, 30))
img_b = mkimg("WIFI PASSWORD: tiger-7781", (30, 30, 200))
out = {}

# 1) generation with image
t = time.time()
r = c.models.generate_content(model="gemini-3.8-flash",
    contents=[types.Part.from_bytes(data=img_a, mime_type="image/png"), "What text is written in this image, and what is the background color? Answer briefly."],
    config=types.GenerateContentConfig(temperature=0))
out["gen_image"] = {"text": r.text, "usage": r.usage_metadata.model_dump(exclude_none=True), "sec": round(time.time()-t, 2)}

# 2) thinking config variants
for lvl in ["minimal", "low"]:
    try:
        t = time.time()
        r = c.models.generate_content(model="gemini-3.8-flash", contents="What is 17*23? Answer with the number only.",
            config=types.GenerateContentConfig(temperature=0, thinking_config=types.ThinkingConfig(thinking_level=lvl)))
        out[f"think_{lvl}"] = {"text": r.text, "usage": r.usage_metadata.model_dump(exclude_none=True), "sec": round(time.time()-t, 2)}
    except Exception as e:
        out[f"think_{lvl}"] = f"ERR {type(e).__name__}: {str(e)[:300]}"

# 3) media resolution
for mr in ["MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_MEDIUM", "MEDIA_RESOLUTION_HIGH"]:
    try:
        r = c.models.generate_content(model="gemini-3.8-flash",
            contents=[types.Part.from_bytes(data=img_b, mime_type="image/png"), "Transcribe the text exactly."],
            config=types.GenerateContentConfig(temperature=0, media_resolution=mr, thinking_config=types.ThinkingConfig(thinking_level="minimal")))
        out[f"mr_{mr}"] = {"text": r.text, "usage": r.usage_metadata.model_dump(exclude_none=True)}
    except Exception as e:
        out[f"mr_{mr}"] = f"ERR {type(e).__name__}: {str(e)[:300]}"

# 4) embeddings
def emb(model, contents, **cfg):
    r = c.models.embed_content(model=model, contents=contents, config=types.EmbedContentConfig(**cfg) if cfg else None)
    return [np.array(e.values) for e in r.embeddings]

for model in ["gemini-embedding-2", "gemini-embedding-001"]:
    try:
        v = emb(model, ["Where did I park my car?", "My car is on parking level B, spot 42.", "The wifi password is tiger-7781."])
        out[f"{model}_text"] = {"dim": len(v[0]), "cos(q,park)": float(v[0]@v[1]/np.linalg.norm(v[0])/np.linalg.norm(v[1])),
                                "cos(q,wifi)": float(v[0]@v[2]/np.linalg.norm(v[0])/np.linalg.norm(v[2]))}
    except Exception as e:
        out[f"{model}_text"] = f"ERR {type(e).__name__}: {str(e)[:300]}"

try:
    vi = emb("gemini-embedding-2", [types.Content(parts=[types.Part.from_bytes(data=img_a, mime_type="image/png")]),
                                    types.Content(parts=[types.Part.from_bytes(data=img_b, mime_type="image/png")])])
    vq = emb("gemini-embedding-2", ["Where did I park my car?", "What is the wifi password?"])
    cs = lambda a, b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
    out["emb2_image"] = {"dim": len(vi[0]), "cos(qpark,imgPark)": cs(vq[0], vi[0]), "cos(qpark,imgWifi)": cs(vq[0], vi[1]),
                         "cos(qwifi,imgPark)": cs(vq[1], vi[0]), "cos(qwifi,imgWifi)": cs(vq[1], vi[1])}
except Exception as e:
    out["emb2_image"] = f"ERR {type(e).__name__}: {str(e)[:500]}"

# 5) interleaved text+image in one embedding, task types, output dims
try:
    v = emb("gemini-embedding-2", [types.Content(parts=[types.Part.from_text(text="Photo I took this morning:"), types.Part.from_bytes(data=img_a, mime_type="image/png")])])
    out["emb2_interleaved"] = {"n": len(v), "dim": len(v[0])}
except Exception as e:
    out["emb2_interleaved"] = f"ERR {type(e).__name__}: {str(e)[:500]}"
for cfg in [{"task_type": "RETRIEVAL_QUERY"}, {"output_dimensionality": 768}]:
    try:
        v = emb("gemini-embedding-2", ["hello world"], **cfg)
        out[f"emb2_cfg_{list(cfg)[0]}"] = {"dim": len(v[0])}
    except Exception as e:
        out[f"emb2_cfg_{list(cfg)[0]}"] = f"ERR {type(e).__name__}: {str(e)[:300]}"

print(json.dumps(out, indent=1, default=str))
json.dump(out, open("results/probe_models.json", "w"), indent=1, default=str)
