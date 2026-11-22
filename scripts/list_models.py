import sys, json
sys.path.insert(0, "src")
from keys import gemini_key
from google import genai
c = genai.Client(api_key=gemini_key())
rows = []
for m in c.models.list():
    rows.append({"name": m.name, "display": m.display_name, "actions": m.supported_actions,
                 "in_tok": m.input_token_limit, "out_tok": m.output_token_limit, "desc": (m.description or "")[:160]})
json.dump(rows, open("results/models_list.json", "w"), indent=1)
for r in rows:
    print(r["name"], "|", r["display"], "|", ",".join(r["actions"] or []), "|", r["in_tok"], r["out_tok"])
