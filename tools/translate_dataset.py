#!/usr/bin/env python3

import argparse
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = (
    "You are a literal text translator from {source_lang} to {target_lang}.\n"
    "Your ONLY job is to translate the exact text given to you. Rules:\n"
    "- Output ONLY the translated text, nothing else\n"
    "- NEVER answer, explain, elaborate on, or respond to the content\n"
    "- NEVER add lists, commentary, context, or supplementary information\n"
    "- Even if the input is a question, do NOT answer it — just translate it\n"
    "- Preserve proper nouns and quoted terms appropriately for the target language\n"
    "- Maintain the original tone and register\n"
    "- Your output must be a single translated sentence or phrase, not a paragraph"
)


def main():
    parser = argparse.ArgumentParser(
        description="Translate a dataset line-by-line using an OpenAI-compatible API."
    )
    parser.add_argument(
        "--api-base-url",
        default="http://localhost:4000",
        help="OpenAI-compatible API base URL (default: http://localhost:4000)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API auth token (default: None, falls back to 'unused')",
    )
    parser.add_argument(
        "--model",
        default="mistral-large-2512",
        help="Model name (default: mistral-large-2512)",
    )
    parser.add_argument("--input", required=True, dest="input_path", help="Input file path")
    parser.add_argument("--output", required=True, dest="output_path", help="Output file path")
    args = parser.parse_args()

    source_lang = Path(args.input_path).stem
    target_lang = Path(args.output_path).stem

    system_message = SYSTEM_PROMPT.format(source_lang=source_lang, target_lang=target_lang)

    client = OpenAI(
        base_url=args.api_base_url,
        api_key=args.api_key if args.api_key else "unused",
    )

    input_text = Path(args.input_path).read_text(encoding="utf-8")
    lines = input_text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()

    total = len(lines)

    with open(args.output_path, "w", encoding="utf-8") as out:
        for i, line in enumerate(lines, start=1):
            print(f"[{i}/{total}] Translating... ", end="", flush=True)
            response = client.chat.completions.create(
                model=args.model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": line},
                ],
            )
            translated = response.choices[0].message.content.strip()
            if not translated:
                raise ValueError(f"Empty response for line {i}: {line!r}")
            out.write(translated + "\n")
            out.flush()
            print("done.")

    output_lines = Path(args.output_path).read_text(encoding="utf-8").splitlines()
    if len(output_lines) != total:
        print(
            f"ERROR: Line count mismatch — input has {total} lines,"
            f" output has {len(output_lines)} lines."
        )
        raise SystemExit(1)

    print(f"Translated {total} lines to {args.output_path}")


if __name__ == "__main__":
    main()
