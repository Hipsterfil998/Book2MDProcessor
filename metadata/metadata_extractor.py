"""metadata_extractor.py — Bibliographic metadata extractor using Qwen2.5 via vLLM.

Uses pre-saved eval pages produced during conversion:
  - Front pages (lowest indices) → author, title, year
  - Body pages (middle indices)  → genre
"""

import json
import pandas as pd
from pathlib import Path
from config import TEXT_MODEL_ID, METADATA_MAX_NEW_TOKENS, BIBLIO_PROMPT, GENRE_PROMPT, ENABLE_PREFIX_CACHING
from vllm import LLM, SamplingParams
from utils import suppress_worker_stderr


class MetadataExtractor:
    """Extracts bibliographic metadata from pre-saved eval pages."""

    def __init__(self, model_id: str = TEXT_MODEL_ID, max_new_tokens: int = METADATA_MAX_NEW_TOKENS):
        self.max_new_tokens = max_new_tokens
        with suppress_worker_stderr():
            self.llm = LLM(model=model_id, dtype="bfloat16",
                           enable_prefix_caching=ENABLE_PREFIX_CACHING)

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
            guaranteed = md_files[:10]   # first 10 pages are always sampled
            front_files = guaranteed[:5]  # pages 0-4 → author/title/year
            body_files  = guaranteed[5:][-3:]  # last 3 of guaranteed → genre
            samples.append({
                "book_name": book_dir.name,
                "front_text": "\n\n".join(f.read_text(encoding="utf-8") for f in front_files),
                "body_text":  "\n\n".join(f.read_text(encoding="utf-8") for f in body_files),
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
        """Collect eval pages → infer biblio + genre → save CSV.

        If output_csv already exists, books already present in it are skipped
        and new records are appended.
        """
        samples = self.collect_samples(output_dir)
        if not samples:
            print("[warn] No eval pages found in", output_dir)
            return pd.DataFrame()

        existing_df = pd.DataFrame()
        if Path(output_csv).exists():
            existing_df = pd.read_csv(output_csv, encoding="utf-8")
            already_done = set(existing_df["book"].dropna())
            samples = [s for s in samples if s["book_name"] not in already_done]
            if not samples:
                print(f"[info] All books already in {output_csv}, nothing to do.")
                return existing_df

        biblio_dataset = [
            [
                {"role": "system", "content": BIBLIO_PROMPT},
                {"role": "user",   "content": f"Book filename: {s['book_name']}\n\n{s['front_text']}"},
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
        _no_think = {"enable_thinking": False}
        biblio_outputs = [o.outputs[0].text for o in self.llm.chat(biblio_dataset, sp, chat_template_kwargs=_no_think)]
        genre_outputs  = [o.outputs[0].text for o in self.llm.chat(genre_dataset,  sp, chat_template_kwargs=_no_think)]

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

        if not existing_df.empty:
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"Saved {len(df)} records to {output_csv}")
        return df
