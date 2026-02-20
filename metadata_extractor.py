"""
Bibliographic metadata extractor using Qwen2.5 via vLLM.
Processes book text extracts and outputs structured author/title/year data.
"""

import json
import pandas as pd
from pathlib import Path
from vllm import LLM, SamplingParams
from huggingface_hub import login


SYSTEM_PROMPT = """\
You are a bibliographic information extraction assistant. Extract author, title, and publication year and genre from texts in German or Italian.
For the genres rifer to the following list:
- Journalistic: newspapers, magazines, newspapers by clubs and associations, publications of the provincial press office, district and municipal gazettes
- Functional/Gebrauchstexte: school books, health guides, cookbooks, hiking guides, regional guides, instruction manuals, advertisements, programme booklets, ordinances, etc.
- Factual/Non fiction/Wissenschaft: non-fiction, popular science essays in journals/yearbooks/conference proceedings, scientific texts, theses, dissertations, thematic periodicals
- Fiction/Belletristik: novels, novellas, short stories, biographies, literary letters, essays, sagas, fairy tales, cityscapes tales, townscapes, regional literature, crime stories, trivial literature, children's literature, calendar literature, anthologies, dramatic anthologies, drama texts, autobiographical literature (diaries, childhood memories diaries, childhood memories, war memoirs, travelogues, reports on mountaineering and expeditions reports)

Output format (JSON):
{{ 
    "author": "Author Name","title": "Work Title","year": "YYYY", "genre": "Genre"
}}

Rules:
- for author and title refer to {full_title}. If they are not presented in the text use the information given remembering that info is separated by _.
- If "year" information is missing, use null
- Keep original language for author and title
- Extract only the publication year (4 digits)
- Handle multiple authors as comma-separated string
- check the output format and correct it if needed
- don't repeat the same json in the output; author, title and year should be unique
- Output only valid JSON, no additional text\
"""


class MetadataExtractor:
    """
    End-to-end pipeline for extracting bibliographic metadata from book text extracts.

    Args:
        model_id: HuggingFace model ID to use for generation.
        hf_token: HuggingFace API token. If None, assumes already logged in.
        max_extract_chars: Max characters to keep per book extract.
        max_new_tokens: Max tokens for model generation.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        hf_token: str | None = None,
        max_extract_chars: int = 1500,
        max_new_tokens: int = 128,
    ):
        self.model_id = model_id
        self.max_extract_chars = max_extract_chars
        self.max_new_tokens = max_new_tokens

        if hf_token:
            login(token=hf_token)

        self.llm = self._load_model()

    def _load_model(self) -> LLM:
        """Initialize the vLLM engine."""
        return LLM(model=self.model_id, dtype="bfloat16")

    def collect_extracts(self, input_dir: str) -> list[tuple[str, str]]:
        """
        Walk input_dir, read .md files, and return (book_name, extract) pairs.

        EPUB markdown uses '\\n\\n---\\n\\n' as chapter separator; only the
        first section is used. PDF markdown has no such separator, so the full
        text is used before truncation.

        Args:
            input_dir: Root directory containing one sub-folder per book.
        Returns:
            List of (title, extract_text) tuples.
        """
        couples = []
        for book_dir in Path(input_dir).iterdir():
            if not book_dir.is_dir():
                continue
            for md_file in book_dir.glob("*.md"):
                text = md_file.read_text(encoding="utf-8")
                first_section = text.split("\n\n---\n\n")[0]
                extract = first_section[: self.max_extract_chars].strip()
                couples.append((book_dir.name, extract))
        return couples

    def _build_dataset(self, couples: list[tuple[str, str]]) -> list:
        """Format (title, extract) pairs into chat messages for the pipeline."""
        return [
            [
                {"role": "system", "content": SYSTEM_PROMPT.format(full_title=title)},
                {"role": "user", "content": extract},
            ]
            for title, extract in couples
        ]

    def _run_inference(self, dataset: list) -> list[str]:
        """Run batched inference and return raw generated strings."""
        sampling_params = SamplingParams(max_tokens=self.max_new_tokens, temperature=0.0)
        outputs = self.llm.chat(dataset, sampling_params=sampling_params)
        return [out.outputs[0].text for out in outputs]

    @staticmethod
    def _parse_output(raw: str, idx: int) -> dict | None:
        """
        Attempt to parse a single model output string into a dict.
        Falls back to extracting the first JSON object on failure.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:-1]).strip()

        # Wrap bare objects into a JSON array so we can handle multiple
        if cleaned.startswith("{"):
            cleaned = "[" + cleaned.replace("}\n{", "},\n{").replace("} {", "},{") + "]"

        try:
            data = json.loads(cleaned)
            return data if isinstance(data, list) else [data]
        except (json.JSONDecodeError, ValueError):
            # Last-resort: grab the first object
            fragment = cleaned.split("\n\n")[0].split("}\n")[0]
            if not fragment.endswith("}"):
                fragment += "}"
            try:
                return [json.loads(fragment)]
            except Exception:
                print(f"[warn] Could not parse output at index {idx}")
                return None

    def _parse_results(self, raw_outputs: list[str]) -> pd.DataFrame:
        """Parse all model outputs into a clean DataFrame."""
        records = []
        for idx, raw in enumerate(raw_outputs):
            parsed = self._parse_output(raw, idx)
            if parsed:
                records.extend(parsed)

        df = pd.DataFrame(records)
        df.replace("null", None, inplace=True)
        df.dropna(how="all", inplace=True)
        return df

    def save(self, df: pd.DataFrame, output_path: str) -> None:
        """Save DataFrame to .xlsx or .csv based on file extension."""
        if output_path.endswith(".xlsx"):
            df.to_excel(output_path, index=False, engine="openpyxl")
        else:
            df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved {len(df)} records to {output_path}")

    def run(self, input_dir: str, output_path: str) -> pd.DataFrame:
        """
        Full pipeline: collect extracts → infer → parse → save.

        Args:
            input_dir: Directory with book sub-folders containing .md files.
            output_path: Output file path (.csv or .xlsx).
        Returns:
            DataFrame with extracted metadata.
        """
        couples = self.collect_extracts(input_dir)
        dataset = self._build_dataset(couples)
        raw_outputs = self._run_inference(dataset)
        df = self._parse_results(raw_outputs)
        self.save(df, output_path)
        return df
