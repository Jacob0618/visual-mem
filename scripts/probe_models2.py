import sys, io, json
sys.path.insert(0, "src")
from keys import gemini_key
from google import genai
from google.genai import types
from PIL import Image, ImageDraw
import numpy as np
c = genai.Client(api_key=gemini_key())
cs = lambda a, b: float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))

def mkimg(text, color, size=(512, 384)):
    im = Image.new("RGB", size, color); d = ImageDraw.Draw(im)
    d.rectangle([40, 120, 470, 260], fill="white"); d.text((60, 170), text, fill="black")
    b = io.BytesIO(); im.save(b, format="PNG"); return b.getvalue()

img_a = mkimg("PARKING LEVEL B, SPOT 42", (200, 30, 30)); img_b = mkimg("WIFI PASSWORD: tiger-7781", (30, 30, 200))
out = {}
txt = ["Where did I park my car?", "My car is on parking level B, spot 42.", "The wifi password is tiger-7781."]
r = c.models.embed_content(model="gemini-embedding-2", contents=[types.Content(parts=[types.Part.from_text(text=t)]) for t in txt])
out["emb2_3contents_n"] = len(r.embeddings)
if len(r.embeddings) == 3:
    v = [np.array(e.values) for e in r.embeddings]
    out["emb2_text"] = {"cos(q,park)": cs(v[0], v[1]), "cos(q,wifi)": cs(v[0], v[2])}
r = c.models.embed_content(model="gemini-embedding-2", contents=[types.Content(parts=[types.Part.from_bytes(data=x, mime_type="image/png")]) for x in [img_a, img_b]])
out["emb2_2images_n"] = len(r.embeddings)
vi = [np.array(e.values) for e in r.embeddings]
rq = c.models.embed_content(model="gemini-embedding-2", contents=[types.Content(parts=[types.Part.from_text(text=t)]) for t in ["Where did I park my car?", "What is the wifi password?"]])
vq = [np.array(e.values) for e in rq.embeddings]
if len(vi) == 2 and len(vq) == 2:
    out["emb2_xmodal"] = {"qpark-imgPark": cs(vq[0], vi[0]), "qpark-imgWifi": cs(vq[0], vi[1]), "qwifi-imgPark": cs(vq[1], vi[0]), "qwifi-imgWifi": cs(vq[1], vi[1])}
# task types on emb2
for tt in ["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "QUESTION_ANSWERING"]:
    try:
        r = c.models.embed_content(model="gemini-embedding-2", contents=[types.Content(parts=[types.Part.from_text(text="Where did I park my car?")])], config=types.EmbedContentConfig(task_type=tt))
        out[f"emb2_tt_{tt}_cos_vs_default"] = cs(np.array(r.embeddings[0].values), vq[0])
    except Exception as e:
        out[f"emb2_tt_{tt}"] = f"ERR {str(e)[:200]}"
# usage/billing metadata
try:
    out["emb2_meta"] = str(r.metadata)[:300]
except Exception as e:
    out["emb2_meta"] = str(e)
# media resolution on generation
for mr in ["MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_MEDIUM", "MEDIA_RESOLUTION_HIGH"]:
    try:
        r = c.models.generate_content(model="gemini-3.8-flash", contents=[types.Part.from_bytes(data=img_b, mime_type="image/png"), "Transcribe the text exactly."],
            config=types.GenerateContentConfig(temperature=0, media_resolution=mr, thinking_config=types.ThinkingConfig(thinking_level="low")))
        u = r.usage_metadata
        out[f"gen_{mr}"] = {"text": r.text, "prompt_tok": u.prompt_token_count, "details": [d.model_dump(exclude_none=True) for d in (u.prompt_tokens_details or [])], "thoughts": u.thoughts_token_count}
    except Exception as e:
        out[f"gen_{mr}"] = f"ERR {str(e)[:300]}"
print(json.dumps(out, indent=1))
json.dump(out, open("results/probe_models2.json", "w"), indent=1)
