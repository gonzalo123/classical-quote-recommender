from __future__ import annotations

import zipfile

from app.ingestion.loader import load_corpus


def test_load_corpus_supports_epub(tmp_path) -> None:
    epub_path = tmp_path / "sample.epub"

    with zipfile.ZipFile(epub_path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
""",
        )
        archive.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0" encoding="UTF-8"?>
<package version="3.0" xmlns="http://www.idpf.org/2007/opf" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>The Odyssey Sample</dc:title>
    <dc:creator>Homer</dc:creator>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="chapter1"/>
  </spine>
</package>
""",
        )
        archive.writestr(
            "OEBPS/chapter1.xhtml",
            """<html xmlns="http://www.w3.org/1999/xhtml">
  <body>
    <section>
      <h1>Book I</h1>
      <p>Tell me, Muse, of the man of many ways.</p>
      <p>He learned the minds of many cities.</p>
    </section>
  </body>
</html>
""",
        )

    documents = load_corpus(tmp_path)

    assert len(documents) == 1
    assert documents[0].author == "Homer"
    assert documents[0].work == "The Odyssey Sample"
    assert "Tell me, Muse" in documents[0].text
    assert documents[0].metadata["corpus_type"] == "epub"
