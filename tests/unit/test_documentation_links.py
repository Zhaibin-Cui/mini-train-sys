import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_LINK = re.compile(r"\[[^\]]*]\(([^)]+)\)")


def _documentation_files() -> list[Path]:
    files = [REPOSITORY_ROOT / "README.md"]
    for directory in ("docs", "reports"):
        files.extend((REPOSITORY_ROOT / directory).rglob("*.md"))
    files.extend((REPOSITORY_ROOT / "experiments").rglob("README.md"))
    files.extend((REPOSITORY_ROOT / "scripts").rglob("*.md"))
    return sorted(files)


class DocumentationLinksTest(unittest.TestCase):
    def test_local_markdown_links_resolve(self) -> None:
        missing = []
        for document in _documentation_files():
            text = document.read_text(encoding="utf-8")
            for match in MARKDOWN_LINK.finditer(text):
                target_text = match.group(1).strip().strip("<>")
                if not target_text or target_text.startswith(
                    ("#", "http://", "https://", "mailto:")
                ):
                    continue
                relative_path = target_text.split("#", 1)[0]
                if not relative_path:
                    continue
                target = (document.parent / relative_path).resolve()
                if not target.exists():
                    line = text.count("\n", 0, match.start()) + 1
                    missing.append(
                        f"{document.relative_to(REPOSITORY_ROOT)}:{line}: "
                        f"{target_text}"
                    )

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
