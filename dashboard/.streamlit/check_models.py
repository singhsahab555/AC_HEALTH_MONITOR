import os
import tomllib
from pathlib import Path

import google.generativeai as genai

# Key comes from GEMINI_API_KEY env var or secrets.toml next to this file — never hard-code it.
_secrets = Path(__file__).with_name("secrets.toml")
_key = os.environ.get("GEMINI_API_KEY") or tomllib.loads(_secrets.read_text(encoding="utf-8")).get("GEMINI_API_KEY", "")
genai.configure(api_key=_key)

print("Models available for your AI Assistant:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"✅ {m.name}")