import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from baby_food_rag.download import download_sources, select_for_download
from baby_food_rag.manifest import load_manifest
from baby_food_rag.types import Source

MANIFEST = Path(__file__).resolve().parents[1] / "sources" / "manifest.yaml"

PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = self.server.body_for(self.path)  # type: ignore[attr-defined]
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def _serve(bodies: dict[str, bytes]):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.body_for = lambda path: bodies.get(path)  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _source(
    source_id: str, url: str, filename: str, sha: str | None, download: bool = True
) -> Source:
    return Source(
        id=source_id,
        title="t",
        organ="o",
        year=2019,
        country="BR",
        url=url,
        license="test",
        role="core",
        download=download,
        filename=filename,
        sha256=sha,
    )


def test_default_download_skips_aap_and_blocked_niaid():
    sources = load_manifest(MANIFEST)
    chosen = select_for_download(sources)
    ids = {source.id for source in chosen}
    assert "AAP-2019" not in ids
    assert "NIAID-2017" not in ids
    assert "MS-2019" in ids
    assert "UY-MSP" in ids


def test_checksum_match_and_mismatch_and_continue(tmp_path):
    good = PDF
    bad_html = b"<html>nope</html>"
    server = _serve({"/good.pdf": good, "/html.pdf": bad_html, "/missing.pdf": None})
    try:
        port = server.server_address[1]
        digest = hashlib.sha256(good).hexdigest()
        sources = [
            _source("OK", f"http://127.0.0.1:{port}/good.pdf", "good.pdf", digest),
            _source("BAD", f"http://127.0.0.1:{port}/good.pdf", "bad.pdf", "0" * 64),
            _source("HTML", f"http://127.0.0.1:{port}/html.pdf", "html.pdf", None),
            _source("MISS", f"http://127.0.0.1:{port}/missing.pdf", "missing.pdf", None),
            _source(
                "AAP", "https://doi.org/10.1542/peds.2019-0281", "aap.pdf", None, download=False
            ),
        ]
        results = download_sources(sources, tmp_path)
    finally:
        server.shutdown()
    by_id = {result.source_id: result for result in results}
    assert by_id["OK"].status == "ok"
    assert (tmp_path / "good.pdf").read_bytes() == good
    assert by_id["OK"].sha256 == hashlib.sha256(good).hexdigest()
    assert by_id["BAD"].status == "failed"
    assert "checksum" in by_id["BAD"].message
    assert not (tmp_path / "bad.pdf").exists()
    assert by_id["HTML"].status == "failed"
    assert by_id["MISS"].status == "failed"
    assert "HTTP 404" in by_id["MISS"].message
    assert by_id["AAP"].status == "skipped"
    assert (tmp_path / "good.pdf").exists()
