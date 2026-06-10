from __future__ import annotations

from collections import defaultdict

from mind.trajectory.stage_a_closeout import build_closeout_family_split


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subset in ("popular", "random", "adversarial"):
        for image_index in range(8):
            image_id = f"shared-{image_index:02d}"
            for model in ("m0", "m1"):
                rows.append(
                    {
                        "model_name": model,
                        "dataset_name": "repope",
                        "source_dataset": "repope",
                        "subset": subset,
                        "sample_id": f"{model}-{subset}-{image_index}",
                        "image_id": image_id,
                        "label": 0,
                        "parsed_answer": 0,
                    }
                )
    return rows


def test_family_level_split_preserves_image_id_grouping() -> None:
    manifest = build_closeout_family_split(
        _rows(),
        family="repope",
        seed=20260506,
    )

    splits_by_image: dict[str, set[str]] = defaultdict(set)
    for row in manifest["assignments"]:
        splits_by_image[str(row["image_id"])].add(str(row["split"]))

    assert manifest["split_scope"] == "repope_family"
    assert set(manifest["counts_per_split"]) == {"encoder_train", "bank", "cal", "test"}
    assert all(len(splits) == 1 for splits in splits_by_image.values())


def test_dash_b_split_uses_all_subset() -> None:
    rows = [
        {
            "model_name": "m0",
            "dataset_name": "dash-b",
            "source_dataset": "dash-b",
            "subset": "all",
            "sample_id": f"s{i}",
            "image_id": f"image-{i}",
            "label": 0,
            "parsed_answer": 0,
        }
        for i in range(10)
    ]

    manifest = build_closeout_family_split(rows, family="dash-b", seed=20260506)

    assert manifest["split_scope"] == "dash-b"
    assert manifest["allowed_subsets"] == ["all"]
    assert manifest["image_id_overlap_validation"]["status"] == "passed"
