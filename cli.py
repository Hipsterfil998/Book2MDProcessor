"""cli.py — Command-line interface for Book2MDProcessor.

Usage examples:
    python cli.py convert --pdf
    python cli.py convert --epub
    python cli.py convert --simple
    python cli.py metadata
    python cli.py evaluate
    python cli.py evaluate --bertscore
    python cli.py parse --langs it de --format conllu
"""

import argparse
from config import INPUT_DIR, OUTPUT_DIR, SCORES_DIR, METADATA_CSV, PARSE_LANGS, PARSE_OUTPUT_FORMAT


def cmd_convert(args):
    from book_converter import ConverterPipeline
    pipeline = ConverterPipeline(input_dir=args.input, output_dir=args.output)
    if args.pdf:
        pipeline.run_pdf_llm()
    elif args.epub:
        pipeline.run_epub_llm()
    elif args.simple:
        pipeline.run_simple()


def cmd_metadata(args):
    from metadata.metadata_extractor import MetadataExtractor
    MetadataExtractor().run(output_dir=args.output, output_csv=args.csv)


def cmd_evaluate(args):
    from quality_evaluation.evaluator import QualityEvaluator
    QualityEvaluator(use_bertscore=args.bertscore).evaluate_all(
        output_dir=args.output, scores_dir=args.scores
    )


def cmd_parse(args):
    from dependency_parsing.dependency_parsing import DependencyParser
    DependencyParser(langs=args.langs, output_format=args.format).run(
        input_dir=args.output, output_dir=args.parsed
    )


def main():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Book2MDProcessor — PDF/EPUB to Markdown pipeline",
    )
    parser.add_argument("--input",   default=INPUT_DIR,    metavar="DIR", help="Input directory (default: %(default)s)")
    parser.add_argument("--output",  default=OUTPUT_DIR,   metavar="DIR", help="Output directory (default: %(default)s)")
    parser.add_argument("--scores",  default=SCORES_DIR,   metavar="DIR", help="Scores directory (default: %(default)s)")
    parser.add_argument("--csv",     default=METADATA_CSV, metavar="FILE", help="Metadata CSV path (default: %(default)s)")
    parser.add_argument("--parsed",  default="parsed/",    metavar="DIR", help="Parsed output directory (default: %(default)s)")

    sub = parser.add_subparsers(dest="command", required=True)

    # convert
    p_conv = sub.add_parser("convert", help="Convert books to Markdown")
    mode = p_conv.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pdf",    action="store_true", help="PDF → Markdown via LLM (Qwen3-VL)")
    mode.add_argument("--epub",   action="store_true", help="EPUB → Markdown via LLM (Qwen3)")
    mode.add_argument("--simple", action="store_true", help="PDF/EPUB → Markdown rule-based (no LLM)")

    # metadata
    sub.add_parser("metadata", help="Extract author/title/year/genre from converted books")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate conversion quality (NED, BLEU, MarkdownF1)")
    p_eval.add_argument("--bertscore", action="store_true", help="Also compute BERTScore (slower)")

    # parse
    p_parse = sub.add_parser("parse", help="Run dependency parsing on converted Markdown")
    p_parse.add_argument("--langs",  nargs="+", default=PARSE_LANGS,         metavar="LANG", help="Languages (default: %(default)s)")
    p_parse.add_argument("--format", default=PARSE_OUTPUT_FORMAT, choices=["conllu", "json", "both"], help="Output format (default: %(default)s)")

    args = parser.parse_args()
    {"convert": cmd_convert, "metadata": cmd_metadata, "evaluate": cmd_evaluate, "parse": cmd_parse}[args.command](args)


if __name__ == "__main__":
    main()
