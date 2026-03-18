"""One-time script: build definitions JSON from Qdrant and save to definitions/.

Usage:
    python scripts/build_definitions.py

Requires a live Qdrant instance with ingested CRR data.
Output: definitions/definitions_en.json and definitions/definitions_it.json
"""
from dotenv import load_dotenv
load_dotenv()

from src.indexing.vector_store import VectorStore
from src.query.definitions_store import DefinitionsStore

vs = VectorStore()
vs.connect()
ds = DefinitionsStore(vs)
for lang in ("en", "it"):
    try:
        ds.load(lang)
        count = len(ds._definitions.get(lang, {}))
        print(f"{lang}: {count} definitions")
    except Exception as exc:
        print(f"{lang}: FAILED — {exc}")
