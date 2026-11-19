"""Download official PDFs listed in the manifest.

Adapted from the catalog ``download.sh``: same user agent, same official
URLs, and a failure on one file does not stop the others. A sha256 in the
manifest is checked. Files that are not PDFs are rejected.

The AAP row has ``download: false`` and no direct PDF URL. A third-party
copy is never fetched.
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from baby_food_rag.types import Source

USER_AGENT = "Mozilla/5.0 (compatible; research-download/1.0)"
_PDF_MAGIC = b"%PDF-"


@dataclass(frozen=True)
class DownloadResult:
    source_id: str
    status: str
    message: str
    sha256: str | None = None
    path: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "skipped"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def select_for_download(sources: list[Source], ids: list[str] | None = None) -> list[Source]:
    if not ids:
        return [source for source in sources if source.download]
    by_id = {source.id: source for source in sources}
    chosen: list[Source] = []
    for source_id in ids:
        if source_id not in by_id:
            raise KeyError(source_id)
        chosen.append(by_id[source_id])
    return chosen


def download_sources(
    sources: list[Source],
    dest: Path,
    *,
    force: bool = False,
    timeout: int = 120,
) -> list[DownloadResult]:
    dest.mkdir(parents=True, exist_ok=True)
    results: list[DownloadResult] = []
    for source in sources:
        if not source.download:
            results.append(
                DownloadResult(
                    source_id=source.id,
                    status="skipped",
                    message=(
                        "download is disabled in the manifest. Third-party copies are not fetched."
                    ),
                )
            )
            continue
        results.append(_download_one(source, dest / source.filename, force=force, timeout=timeout))
    return results


def _download_one(source: Source, path: Path, *, force: bool, timeout: int) -> DownloadResult:
    if path.exists() and not force:
        existing = _check_existing(source, path)
        if existing is not None:
            return existing
    partial = path.with_name(path.name + ".part")
    try:
        _fetch(source.url, partial, timeout=timeout)
        if not _is_pdf(partial):
            head = partial.read_bytes()[:40]
            partial.unlink(missing_ok=True)
            return DownloadResult(
                source_id=source.id,
                status="failed",
                message=f"response is not a PDF (starts with {head!r})",
            )
        digest = sha256_file(partial)
        if source.sha256 and digest != source.sha256:
            partial.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            return DownloadResult(
                source_id=source.id,
                status="failed",
                message=f"checksum mismatch: got {digest}, manifest has {source.sha256}",
                sha256=digest,
            )
        partial.replace(path)
    except DownloadFailure as exc:
        partial.unlink(missing_ok=True)
        return DownloadResult(source_id=source.id, status="failed", message=str(exc))
    except Exception as exc:
        partial.unlink(missing_ok=True)
        return DownloadResult(
            source_id=source.id,
            status="failed",
            message=f"{type(exc).__name__}: {exc}",
        )
    note = "checksum matched" if source.sha256 else "no checksum in the manifest; recorded"
    return DownloadResult(
        source_id=source.id,
        status="ok",
        message=f"{note}; {path.stat().st_size} bytes",
        sha256=digest,
        path=str(path),
    )


def _check_existing(source: Source, path: Path) -> DownloadResult | None:
    if not _is_pdf(path):
        return None
    digest = sha256_file(path)
    if source.sha256 and digest != source.sha256:
        return None
    note = (
        "checksum matched" if source.sha256 else "already downloaded; no checksum in the manifest"
    )
    return DownloadResult(
        source_id=source.id,
        status="ok",
        message=f"{note}; kept existing file",
        sha256=digest,
        path=str(path),
    )


class DownloadFailure(Exception):
    pass


def _is_pdf(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(len(_PDF_MAGIC)) == _PDF_MAGIC


def _fetch(url: str, dest: Path, *, timeout: int) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, dest.open("wb") as out:
            while True:
                block = response.read(1 << 16)
                if not block:
                    break
                out.write(block)
    except urllib.error.HTTPError as exc:
        raise DownloadFailure(f"HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise DownloadFailure(f"could not fetch {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise DownloadFailure(f"timed out fetching {url}") from exc
