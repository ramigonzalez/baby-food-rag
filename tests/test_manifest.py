from pathlib import Path

from baby_food_rag.manifest import default_index_sources, load_manifest, sources_for_index

MANIFEST = Path(__file__).resolve().parents[1] / "sources" / "manifest.yaml"

CORE = {"MS-2019", "ASBAI-SBP-2025", "SBP-MAN-2024", "SBP-BLW-2017"}
CONTRAST = {
    "ESPGHAN-2017",
    "NIAID-2017",
    "NIAID-PARENT-2017",
    "ASCIA-2026",
    "ASCIA-2026-FULL",
    "AAP-2019",
}


def test_manifest_loads_required_fields():
    sources = load_manifest(MANIFEST)
    assert sources
    for source in sources:
        assert source.id
        assert source.title
        assert source.organ
        assert source.country
        assert source.url.startswith("https://")
        assert source.license
        assert source.role
        assert source.filename.endswith(".pdf")


def test_core_is_the_default_index_and_uruguay_is_optional():
    sources = load_manifest(MANIFEST)
    default = default_index_sources(sources)
    assert {source.id for source in default} == CORE
    assert all(source.country == "BR" and source.in_default_index for source in default)
    uruguay = next(source for source in sources if source.id == "UY-MSP")
    assert uruguay.country == "UY"
    assert uruguay.role == "optional"
    assert uruguay not in default
    indexed = sources_for_index(sources, {"optional"})
    assert "UY-MSP" in {source.id for source in indexed}
    assert CONTRAST.isdisjoint({source.id for source in indexed})


def test_contrast_is_flagged_and_out_of_the_default_index():
    sources = load_manifest(MANIFEST)
    by_id = {source.id: source for source in sources}
    for source_id in CONTRAST:
        assert by_id[source_id].role == "contrast"
        assert not by_id[source_id].in_default_index
    default_ids = {source.id for source in default_index_sources(sources)}
    assert CONTRAST.isdisjoint(default_ids)


def test_aap_is_not_downloaded_and_has_no_third_party_url():
    sources = load_manifest(MANIFEST)
    aap = next(source for source in sources if source.id == "AAP-2019")
    assert aap.download is False
    assert aap.url == "https://doi.org/10.1542/peds.2019-0281"
    blob = MANIFEST.read_text(encoding="utf-8").lower()
    assert "allergolyon" not in blob
    assert "third-party" in aap.notes.lower() or "third-party" in blob


def test_core_checksums_are_pinned():
    sources = load_manifest(MANIFEST)
    for source in default_index_sources(sources):
        assert source.sha256 is not None
        assert len(source.sha256) == 64
