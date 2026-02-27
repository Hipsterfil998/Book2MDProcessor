"""
metadata_extractor.py — Bibliographic metadata extractor using Qwen2.5 via vLLM.

Uses pre-saved eval pages produced during conversion:
  - Front pages (lowest indices) → author, title, year
  - Body pages (middle indices)  → genre
"""

import json
import pandas as pd
from pathlib import Path
from vllm import LLM, SamplingParams


BIBLIO_PROMPT = """\
You are a bibliographic information extraction assistant. Extract author, title, and \
publication year from texts in German or Italian.

Output format (JSON):
{{"author": "Author Name", "title": "Work Title", "year": "YYYY"}}

Rules:
- For author and title refer to {full_title}. If not present in the text, use the \
filename info (fields separated by _).
- If year is missing, use null.
- Keep original language for author and title.
- Extract only the 4-digit publication year.
- Multiple authors as comma-separated string.
- Output only valid JSON, no additional text.\
"""

GENRE_PROMPT = """\
You are a literary genre classifier for Italian and German texts.
Based on the provided excerpt from the body of a book, classify it into one of these genres:
- Journalistic: newspapers, magazines, club publications, press office publications, \
district and municipal gazettes
- Functional/Gebrauchstexte: school books, health guides, cookbooks, hiking guides, \
regional guides, manuals, advertisements, programme booklets, ordinances
- Factual/Non fiction/Wissenschaft: non-fiction, popular science, essays in journals \
or conference proceedings, scientific texts, theses, dissertations, thematic periodicals
- Fiction/Belletristik: novels, novellas, short stories, biographies, literary letters, \
essays, sagas, fairy tales, crime stories, children's literature, drama, autobiographical \
literature, travelogues

Output format (JSON):
{{"genre": "Genre"}}

Rules:
- Output only valid JSON, no additional text.\
"""


class MetadataExtractor:
    """
    Extracts bibliographic metadata from pre-saved eval pages.

    Args:
        model_id: HuggingFace model ID for generation.
        max_new_tokens: Max tokens for each generation call.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens: int = 128,
    ):
        self.max_new_tokens = max_new_tokens
        self.llm = LLM(model=model_id, dtype="bfloat16")

    def collect_samples(self, output_dir: str) -> list[dict]:
        """
        Walk output_dir and collect eval text per book.

        For each book subfolder, reads .md files from eval_pages/ or eval_chunks/,
        sorted by page/chunk index:
          - front_text: first 3 pages (title page, TOC, preface area)
          - body_text:  up to 2 middle pages (main content, used for genre)

        Returns:
            List of dicts with keys: book_name, front_text, body_text.
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

            front_text = "\n\n".join(f.read_text(encoding="utf-8") for f in front_files)
            body_text  = "\n\n".join(f.read_text(encoding="utf-8") for f in body_files[:2])

            samples.append({
                "book_name": book_dir.name,
                "front_text": front_text,
                "body_text": body_text,
            })
        return samples

    def _run_inference(self, dataset: list) -> list[str]:
        sampling_params = SamplingParams(max_tokens=self.max_new_tokens, temperature=0.0)
        outputs = self.llm.chat(dataset, sampling_params=sampling_params)
        return [out.outputs[0].text for out in outputs]

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
        """
        Full pipeline: collect eval pages → infer biblio → infer genre → save CSV.

        Args:
            output_dir: Directory with book subfolders (each containing eval_pages/ or eval_chunks/).
            output_csv: Output CSV file path.
        Returns:
            DataFrame with columns: author, title, year, genre, book.
        """
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

        biblio_outputs = self._run_inference(biblio_dataset)
        genre_outputs  = self._run_inference(genre_dataset)

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
