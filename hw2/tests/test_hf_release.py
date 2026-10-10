import json

from hf_release import HW2_DIR, build_release, render_model_card

RESULTS_DIR = HW2_DIR / "results"


def test_card_starts_with_front_matter_for_base_model():
    card = render_model_card(RESULTS_DIR)
    assert card.startswith("---\n") and "base_model: HuggingFaceTB/SmolLM-360M" in card


def test_card_reports_perplexity_from_results():
    perplexity = json.loads((RESULTS_DIR / "eval" / "drope_step1000_s2.json").read_text())["perplexity"]
    assert f"{perplexity:.2f}" in render_model_card(RESULTS_DIR)


def test_release_config_has_no_auto_map(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "config.json").write_text(json.dumps({"model_type": "llama_drope", "auto_map": {"AutoConfig": "drope.X"}}))
    release_dir = build_release(snapshot, RESULTS_DIR, tmp_path / "release")
    assert "auto_map" not in json.loads((release_dir / "config.json").read_text())


def test_release_contains_exactly_weights_config_and_card(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "config.json").write_text("{}")
    release_dir = build_release(snapshot, RESULTS_DIR, tmp_path / "release")
    assert sorted(path.name for path in release_dir.iterdir()) == ["README.md", "config.json", "model.safetensors"]
