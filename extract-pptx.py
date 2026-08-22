"""
Extract slide text and speaker notes from a .pptx without external libraries.

A .pptx is a zip of XML. Visible text lives in <a:t> runs inside
ppt/slides/slideN.xml; speaker notes live in ppt/notesSlides/notesSlideN.xml.
Slides must be sorted numerically — zip order and lexical order both put
slide10 before slide2.
"""
import re
import sys
import zipfile
from html import unescape

TEXT_RUN = re.compile(r"<a:t>(.*?)</a:t>", re.DOTALL)
# Each <a:p> is a paragraph — used to keep bullets on separate lines.
PARA_SPLIT = re.compile(r"</a:p>")
SLIDE_NUM = re.compile(r"slide(\d+)\.xml$")


def runs_to_lines(xml: str) -> list[str]:
    """Group text runs by paragraph so bullets don't collapse into one blob."""
    lines = []
    for para in PARA_SPLIT.split(xml):
        text = "".join(TEXT_RUN.findall(para))
        text = unescape(text).strip()
        if text:
            lines.append(text)
    return lines


def main(path: str) -> None:
    # Decks contain glyphs (✓, ✗, arrows) that Windows' default cp1252 stdout
    # cannot encode; force UTF-8 rather than losing slides to an encode error.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    with zipfile.ZipFile(path) as z:
        names = z.namelist()

        slides = sorted(
            (n for n in names if SLIDE_NUM.search(n) and n.startswith("ppt/slides/")),
            key=lambda n: int(SLIDE_NUM.search(n).group(1)),
        )
        notes = {
            int(SLIDE_NUM.search(n).group(1)): n
            for n in names
            if n.startswith("ppt/notesSlides/notesSlide") and SLIDE_NUM.search(n)
        }

        print(f"# {path}")
        print(f"# {len(slides)} slides, {len(notes)} with speaker notes\n")

        for slide_path in slides:
            num = int(SLIDE_NUM.search(slide_path).group(1))
            xml = z.read(slide_path).decode("utf-8", errors="replace")
            print(f"\n{'=' * 70}\nSLIDE {num}\n{'=' * 70}")
            for line in runs_to_lines(xml):
                print(f"  {line}")

            if num in notes:
                note_xml = z.read(notes[num]).decode("utf-8", errors="replace")
                note_lines = runs_to_lines(note_xml)
                # PowerPoint stores the slide-number placeholder in notes too;
                # drop a lone numeric line that just repeats the slide number.
                note_lines = [l for l in note_lines if l != str(num)]
                if note_lines:
                    print("  --- SPEAKER NOTES ---")
                    for line in note_lines:
                        print(f"    {line}")


if __name__ == "__main__":
    main(sys.argv[1])
