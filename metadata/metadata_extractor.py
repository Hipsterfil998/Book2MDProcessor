"""metadata_extractor.py — Bibliographic metadata extractor using Qwen2.5 via vLLM.

Uses pre-saved eval pages produced during conversion:
  - Front pages (lowest indices) → author, title, year
  - Body pages (middle indices)  → genre
"""

import json
import pandas as pd
from pathlib import Path
from config import TEXT_MODEL_ID, METADATA_MAX_NEW_TOKENS, BIBLIO_PROMPT, GENRE_PROMPT
from vllm import LLM, SamplingParams
from utils import suppress_worker_stderr


class MetadataExtractor:
    """Extracts bibliographic metadata from pre-saved eval pages."""

    def __init__(self, model_id: str = TEXT_MODEL_ID, max_new_tokens: int = METADATA_MAX_NEW_TOKENS):
        self.max_new_tokens = max_new_tokens
        with suppress_worker_stderr():
            self.llm = LLM(model=model_id, dtype="bfloat16")

    def collect_samples(self, output_dir: str) -> list[dict]:
        """Walk output_dir and collect eval text per book.

        Returns list of dicts with keys: book_name, front_text, body_text.
        """
        samples = []
        for book_dir in sorted(Path(output_dir).iterdir()):
            if not book_dir.is_dir():
                continue
            eval_dir = book_dir / "eval_pages"
            if not eval_dir.exists():
                eval_dir = book_dir / "eval_chunks"
            if not eval_dir.exists():
                continue
            md_files = sorted(eval_dir.glob("*.md"), key=lambda p: int(p.stem))
            if not md_files:
                continue
            front_files = md_files[:3]
            body_files = md_files[3:-2] if len(md_files) > 5 else md_files[1:-1]
            samples.append({
                "book_name": book_dir.name,
                "front_text": "\n\n".join(f.read_text(encoding="utf-8") for f in front_files),
                "body_text":  "\n\n".join(f.read_text(encoding="utf-8") for f in body_files[:2]),
            })
        return samples

    @staticmethod
    def _parse_json(raw: str) -> dict | None:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:-1]).strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            fragment = cleaned.split("\n\n")[0].split("}\n")[0]
            if not fragment.endswith("}"):
                fragment += "}"
            try:
                return json.loads(fragment)
            except Exception:
                return None

    def run(self, output_dir: str, output_csv: str) -> pd.DataFrame:
        """Collect eval pages → infer biblio + genre → save CSV."""
        samples = self.collect_samples(output_dir)
        if not samples:
            print("[warn] No eval pages found in", output_dir)
            return pd.DataFrame()

        biblio_dataset = [
            [
                {"role": "system", "content": BIBLIO_PROMPT.format(full_title=s["book_name"])},
                {"role": "user",   "content": s["front_text"]},
            ]
            for s in samples
        ]
        genre_dataset = [
            [
                {"role": "system", "content": GENRE_PROMPT},
                {"role": "user",   "content": s["body_text"]},
            ]
            for s in samples
        ]

        sp = SamplingParams(max_tokens=self.max_new_tokens, temperature=0.0)
        biblio_outputs = [o.outputs[0].text for o in self.llm.chat(biblio_dataset, sp)]
        genre_outputs  = [o.outputs[0].text for o in self.llm.chat(genre_dataset, sp)]

        records = []
        for s, biblio_raw, genre_raw in zip(samples, biblio_outputs, genre_outputs):
            biblio = self._parse_json(biblio_raw) or {}
            genre  = (self._parse_json(genre_raw) or {}).get("genre")
            records.append({
                "book":   s["book_name"],
                "author": biblio.get("author"),
                "title":  biblio.get("title"),
                "year":   biblio.get("year"),
                "genre":  genre,
            })

        df = pd.DataFrame(records)
        df.replace("null", None, inplace=True)
        df.dropna(how="all", inplace=True)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"Saved {len(df)} records to {output_csv}")
        return df
