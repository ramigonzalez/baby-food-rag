from pathlib import Path

import yaml

from baby_food_rag.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources" / "manifest.yaml"


def test_phase2_questions_point_at_known_sources():
    data = yaml.safe_load((ROOT / "eval" / "questions.yaml").read_text(encoding="utf-8"))
    questions = data["questions"]
    assert len(questions) == 15
    known = {source.id for source in load_manifest(MANIFEST)}
    abstain = [item for item in questions if item.get("abstain")]
    assert len(abstain) == 3
    for item in questions:
        assert item["question"].strip()
        for expect in item["expects"]:
            assert expect["source_id"] in known
    reais = yaml.safe_load((ROOT / "eval" / "perguntas_reais.yaml").read_text(encoding="utf-8"))
    assert len(reais["questions"]) == 10


def test_readme_opens_with_the_medical_warning():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("**Not medical advice.**")


def test_gitignore_keeps_pdfs_out_of_git():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/raw/" in text
    assert "*.pdf" in text
