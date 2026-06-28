"""
pdf_to_markdown.py
Batch-convert PDFs to Markdown using Microsoft's markitdown library.
Keeps original PDFs — writes .md files alongside them.
Run: python scripts/utils/pdf_to_markdown.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from markitdown import MarkItDown


def main():
    md = MarkItDown()
    pdf_files = sorted(REPO.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    ok, fail = 0, 0
    for pdf_path in pdf_files:
        # Skip if .md already exists and is newer
        md_path = pdf_path.with_suffix(".md")
        if md_path.exists() and md_path.stat().st_mtime > pdf_path.stat().st_mtime:
            print(f"  [skip] {pdf_path.relative_to(REPO)} (md up-to-date)")
            ok += 1
            continue

        try:
            print(f"  [conv] {pdf_path.relative_to(REPO)} ...", end=" ", flush=True)
            result = md.convert(str(pdf_path))
            text = result.text_content
            if text.strip():
                # Add a source header
                header = f"<!-- Converted from {pdf_path.relative_to(REPO)} with markitdown -->\n\n"
                md_path.write_text(header + text, encoding="utf-8")
                print(f"OK ({len(text)} chars)")
                ok += 1
            else:
                print("WARN: empty output")
                fail += 1
        except Exception as e:
            print(f"FAIL: {e}")
            fail += 1

    print(f"\nDone: {ok} converted/skipped, {fail} failed")


if __name__ == "__main__":
    main()
