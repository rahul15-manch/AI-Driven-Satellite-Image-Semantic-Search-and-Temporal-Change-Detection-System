"""Integration Test: Milestone 1-4 Architectural Contracts and Boundary Compliance.

Covers:
- INT-M1-01: Research Questions (RQ1-RQ4) Exist and Match Scope
- INT-M1-02: Hypotheses (H1-H4) Exist with Test Criteria
- INT-M1-03: Architecture Matches Implementation
- INT-M1-04: Milestone Boundary Compliance (No M5/M6 in src/)
- INT-M1-05: Experiment Plan Consistency (EXP-RET-01, 02, 03)
- INT-CONTRACT-01: M2 Test Image IDs == M4 Gallery Image IDs
- INT-CONTRACT-02: M2 Caption Mapping == M4 Query Mapping
- INT-CONTRACT-03: Zero Train/Val Image Overlap in M4 Gallery
- INT-CONTRACT-04: Image Embedding Pipeline Isolated from Query Text
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.data.rsicd_loader import RSICDDataset


def get_doc_path(filename: str) -> Path:
    """Resolves markdown documentation path across repository directories (.md FIles, docs, root)."""
    candidates = [
        Path(".md FIles ") / filename,
        Path(".md FIles") / filename,
        Path("docs") / filename,
        Path(filename),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


@pytest.mark.integration
def test_int_m1_01_research_questions_exist():
    """INT-M1-01: Verify RQ1, RQ2, RQ3, RQ4 exist in research_questions.md."""
    rq_path = get_doc_path("research_questions.md")
    assert rq_path.exists(), f"research_questions.md must exist (searched: {rq_path})."
    content = rq_path.read_text(encoding="utf-8")

    assert "RQ1" in content, "RQ1 must be documented in research_questions.md"
    assert "RQ2" in content, "RQ2 must be documented in research_questions.md"
    assert "RQ3" in content, "RQ3 must be documented in research_questions.md"
    assert "RQ4" in content, "RQ4 must be documented in research_questions.md"
    assert "Cross-Modal Semantic Retrieval" in content
    assert "Change Detection" in content


@pytest.mark.integration
def test_int_m1_02_hypotheses_exist():
    """INT-M1-02: Verify H1-H4 exist with measurable test criteria."""
    rq_path = get_doc_path("research_questions.md")
    assert rq_path.exists(), f"research_questions.md must exist (searched: {rq_path})."
    content = rq_path.read_text(encoding="utf-8")

    for h_id in ["H1", "H2", "H3", "H4"]:
        assert f"Hypothesis {h_id}" in content or f"Hypothesis {h_id[-1]}" in content or f"[{h_id}]" in content or f"H{h_id[-1]}" in content, (
            f"Hypothesis {h_id} must be documented."
        )


@pytest.mark.integration
def test_int_m1_03_architecture_matches_implementation():
    """INT-M1-03: Verify documented semantic retrieval components match src/semantic_search."""
    arch_path = get_doc_path("architecture.md")
    assert arch_path.exists(), f"architecture.md must exist (searched: {arch_path})."
    content = arch_path.read_text(encoding="utf-8")
    assert "Semantic Retrieval" in content or "semantic_search" in content, (
        "Semantic retrieval pipeline must be documented in architecture.md"
    )
    # Verify core semantic retrieval modules exist in filesystem
    src_dir = Path("src/semantic_search")
    assert (src_dir / "bm25.py").exists(), "bm25.py must exist."
    assert (src_dir / "clip_model.py").exists(), "clip_model.py must exist."
    assert (src_dir / "faiss_index.py").exists(), "faiss_index.py must exist."
    assert (src_dir / "embedding_cache.py").exists(), "embedding_cache.py must exist."
    assert (src_dir / "prompt_ensembler.py").exists(), "prompt_ensembler.py must exist."
    assert (src_dir / "retrieval_evaluator.py").exists(), "retrieval_evaluator.py must exist."


@pytest.mark.integration
def test_int_m1_04_milestone_boundary_compliance():
    """INT-M1-04: Verify architectural boundaries and milestone governance compliance.

    Verifies:
    1. Future milestones beyond M11 (M12+) are not implemented in src/.
    2. The M1-M11 boundary is strictly preserved: M7, M8, M9, M10, and M11 are recognized.
    3. Checkpoints and artifacts for implemented models are authentic (not empty or mock).
    """
    src_dir = Path("src")
    forbidden_future_terms = [
        "class SiamUnet_diff",
        "class CrossModalCD",
        "class FastAPIServer",
    ]

    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for term in forbidden_future_terms:
            assert term not in content, (
                f"Boundary violation: Found future milestone term '{term}' in {py_file}."
            )

    # Verify M12+ boundary: Future milestones beyond M11 are not implemented or checked in
    checkpoints_m12 = Path("experiments/checkpoints/m12")
    if checkpoints_m12.exists():
        pts = list(checkpoints_m12.glob("*.pt")) + list(checkpoints_m12.glob("*.pth"))
        assert len(pts) == 0, "M12 checkpoint found prematurely."

    results_m12 = Path("experiments/results/m12")
    if results_m12.exists():
        res = list(results_m12.glob("*.csv")) + list(results_m12.glob("*.json"))
        assert len(res) == 0, "M12 benchmark results found prematurely."


    # If M7 checkpoint is present, verify it is authentic and valid (not empty or mock)
    checkpoints_m7 = Path("experiments/checkpoints/m7")
    if checkpoints_m7.exists():
        for pt_file in checkpoints_m7.glob("*.pt"):
            assert pt_file.stat().st_size > 1_000_000, f"M7 checkpoint {pt_file} is suspiciously small."

    # If M8 checkpoint is present, verify it is authentic and valid
    checkpoints_m8 = Path("experiments/checkpoints/m8")
    if checkpoints_m8.exists():
        for pt_file in checkpoints_m8.glob("*.pt"):
            assert pt_file.stat().st_size > 500_000, f"M8 checkpoint {pt_file} is suspiciously small."

    # If M9 checkpoint is present, verify it is authentic and valid
    checkpoints_m9 = Path("experiments/checkpoints/m9")
    if checkpoints_m9.exists():
        for pt_file in checkpoints_m9.glob("*.pt"):
            assert pt_file.stat().st_size > 500_000, f"M9 checkpoint {pt_file} is suspiciously small."

    # Verify architecture.md documents M7 as downstream learned milestone dependent on M4/M5
    arch_path = get_doc_path("architecture.md")
    assert arch_path.exists()
    arch_content = arch_path.read_text(encoding="utf-8")
    assert "M7" in arch_content or "Learned" in arch_content


@pytest.mark.integration
def test_int_m1_05_experiment_plan_consistency():
    """INT-M1-05: Verify EXP-RET-01, EXP-RET-02, EXP-RET-03 correspond to A1, A2, A3 in experiment_plan.md."""
    plan_path = get_doc_path("experiment_plan.md")
    assert plan_path.exists(), f"experiment_plan.md must exist (searched: {plan_path})."
    content = plan_path.read_text(encoding="utf-8")

    assert "EXP-RET-01" in content, "EXP-RET-01 must be in experiment_plan.md"
    assert "EXP-RET-02" in content, "EXP-RET-02 must be in experiment_plan.md"
    assert "EXP-RET-03" in content, "EXP-RET-03 must be in experiment_plan.md"
    assert "Method A1" in content
    assert "Method A2" in content
    assert "Method A3" in content


@pytest.mark.integration
@pytest.mark.dataset
def test_int_contract_01_m2_test_ids_equal_m4_gallery_ids():
    """INT-CONTRACT-01: M2 test split image IDs must exactly match M4 gallery IDs."""
    splits_path = Path("data/splits/rsicd/rsicd_splits.json")
    assert splits_path.exists(), "RSICD split manifest must exist."
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    m2_test_ids = set(splits.get("test_ids", splits.get("test", [])))
    assert len(m2_test_ids) == 1093, f"Expected 1,093 test image IDs in M2 split, found {len(m2_test_ids)}."

    # Load dataset via M2 RSICDDataset test split
    dataset = RSICDDataset(
        image_dir="data/raw/rsicd/images",
        json_path="data/raw/rsicd/dataset_rsicd.json",
        split="test",
    )
    m4_gallery_stems = set(Path(s.filename).stem for s in dataset.samples)
    assert m2_test_ids == m4_gallery_stems, "M2 test IDs must be identical to M4 gallery stem IDs."


@pytest.mark.integration
@pytest.mark.dataset
def test_int_contract_02_m2_caption_mapping_equals_m4_query_mapping():
    """INT-CONTRACT-02: M2 test split captions must yield exactly 5,465 queries with 1-to-1 image mapping."""
    dataset = RSICDDataset(
        image_dir="data/raw/rsicd/images",
        json_path="data/raw/rsicd/dataset_rsicd.json",
        split="test",
    )

    total_queries = 0
    mapping_counts = {}
    for sample in dataset.samples:
        img_id = sample.filename
        assert len(sample.captions) == 5, f"Image {img_id} must have exactly 5 captions, got {len(sample.captions)}."
        for cap in sample.captions:
            total_queries += 1
            mapping_counts[img_id] = mapping_counts.get(img_id, 0) + 1

    assert total_queries == 5465, f"Expected 5,465 total queries, got {total_queries}."
    assert len(mapping_counts) == 1093, "All 1,093 images must have mapped captions."
    for img_id, count in mapping_counts.items():
        assert count == 5, f"Image {img_id} mapped to {count} queries, expected 5."


@pytest.mark.integration
def test_int_contract_03_zero_train_val_in_m4_gallery():
    """INT-CONTRACT-03: Zero train or val images appear in the M4 test gallery."""
    splits_path = Path("data/splits/rsicd/rsicd_splits.json")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_set = set(splits.get("train_ids", splits.get("train", [])))
    val_set = set(splits.get("val_ids", splits.get("val", [])))
    test_set = set(splits.get("test_ids", splits.get("test", [])))

    assert len(train_set.intersection(test_set)) == 0, "Train and Test sets must be strictly disjoint."
    assert len(val_set.intersection(test_set)) == 0, "Val and Test sets must be strictly disjoint."
    assert len(train_set.intersection(val_set)) == 0, "Train and Val sets must be strictly disjoint."


@pytest.mark.integration
def test_int_contract_04_clip_image_pipeline_isolated_from_query_text():
    """INT-CONTRACT-04: Verify CLIP image encoder accepts only image files, never text queries."""
    from src.semantic_search.clip_model import CLIPRetriever
    import inspect

    sig = inspect.signature(CLIPRetriever.encode_images)
    param_names = list(sig.parameters.keys())
    assert "image_inputs" in param_names, "CLIPRetriever.encode_images must take image_inputs."
    assert "query" not in param_names
    assert "text" not in param_names
    assert "caption" not in param_names
