"""Load and check sources/manifest.yaml."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from baby_food_rag.types import Source

ROLES = ("core", "optional", "contrast", "reference", "superseded")
REQUIRED = (
    "id",
    "title",
    "organ",
    "year",
    "country",
    "url",
    "license",
    "role",
    "download",
    "filename",
)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_COUNTRY = re.compile(r"^[A-Z]{2,3}$")


def find_manifest(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        path = candidate / "sources" / "manifest.yaml"
        if path.is_file():
            return path
    raise FileNotFoundError("sources/manifest.yaml not found from the working directory")


def load_manifest(path: Path | None = None) -> list[Source]:
    manifest = path or find_manifest()
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
        raise ValueError(f"{manifest} must contain a sources list")
    sources = [_source(item, manifest) for item in raw["sources"]]
    _check_unique(sources)
    return sources


def default_index_sources(sources: list[Source]) -> list[Source]:
    """Brazilian core only. Optional (UY) and contrast stay out."""
    return [source for source in sources if source.role == "core"]


def sources_for_index(sources: list[Source], include: set[str] | None = None) -> list[Source]:
    extra = include or set()
    unknown = extra - {"optional", "contrast", "reference"}
    if unknown:
        raise ValueError(f"Unknown index group: {sorted(unknown)}")
    roles = {"core"} | extra
    return [source for source in sources if source.role in roles]


def _source(item: object, path: Path) -> Source:
    if not isinstance(item, dict):
        raise ValueError(f"{path} has a source that is not a mapping")
    missing = [key for key in REQUIRED if key not in item]
    if missing:
        raise ValueError(f"Source is missing {missing}")
    role = str(item["role"])
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r} on {item.get('id')}")
    year = item["year"]
    if year is not None and not isinstance(year, int):
        raise ValueError(f"{item.get('id')} year must be an integer or null")
    country = str(item["country"])
    if not _COUNTRY.match(country):
        raise ValueError(f"{item.get('id')} country must be a 2-3 letter code")
    url = str(item["url"])
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"{item.get('id')} url must be http(s)")
    filename = str(item["filename"])
    if "/" in filename or filename.startswith("."):
        raise ValueError(f"{item.get('id')} filename must be a plain file name")
    sha = item.get("sha256")
    if sha is not None:
        sha = str(sha)
        if not _SHA.match(sha):
            raise ValueError(f"{item.get('id')} sha256 must be 64 hex characters")
    pages = item.get("pages_expected")
    if pages is not None and not isinstance(pages, int):
        raise ValueError(f"{item.get('id')} pages_expected must be an integer")
    return Source(
        id=str(item["id"]),
        title=str(item["title"]).strip(),
        organ=str(item["organ"]).strip(),
        year=year,
        country=country,
        url=url,
        license=str(item["license"]).strip(),
        role=role,
        download=bool(item["download"]),
        filename=filename,
        language=str(item.get("language") or ""),
        sha256=sha,
        pages_expected=pages,
        notes=str(item.get("notes") or "").strip(),
    )


def _check_unique(sources: list[Source]) -> None:
    ids: set[str] = set()
    names: set[str] = set()
    for source in sources:
        if source.id in ids:
            raise ValueError(f"Duplicate source id {source.id}")
        if source.filename in names:
            raise ValueError(f"Duplicate filename {source.filename}")
        ids.add(source.id)
        names.add(source.filename)
