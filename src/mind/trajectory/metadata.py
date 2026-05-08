"""Column metadata for Stage 0 trajectory audits."""

from __future__ import annotations

POPE_STYLE_DATASET_NAMES = ("pope", "repope")
KNOWN_DATASET_NAMES = (*POPE_STYLE_DATASET_NAMES, "dash-b")
KNOWN_SUBSETS = ("popular", "random", "adversarial")
DASH_B_DATASET_NAME = "dash-b"
DASH_B_SUBSET = "all"
DASH_B_LEGACY_SUBSET = "main"
DATASET_SUBSETS = {
    "pope": KNOWN_SUBSETS,
    "repope": KNOWN_SUBSETS,
    DASH_B_DATASET_NAME: (DASH_B_SUBSET,),
}
UNKNOWN_OBJECT_NAME = "unknown"

DATASET_AUDIT_COLUMNS = (
    "dataset_name",
    "subset",
    "path",
    "status",
    "num_records",
    "num_label_yes",
    "num_label_no",
    "num_missing_image_path",
    "num_missing_question",
    "num_unknown_object",
    "unique_objects",
    "unique_images",
    "num_duplicate_sample_id",
    "num_null_required_fields",
    "num_invalid_label",
)

LABEL_BALANCE_COLUMNS = (
    "dataset_name",
    "subset",
    "status",
    "num_records",
    "num_gt_yes",
    "num_gt_no",
    "num_invalid_label",
    "num_parsed_yes",
    "num_parsed_no",
    "num_parsed_none",
    "num_correct",
    "num_hallucination",
    "hallucination_rate",
    "parsed_answer_status",
)

CACHE_LABEL_BALANCE_COLUMNS = (
    "model_name",
    "dataset_name",
    "subset",
    "num_entries",
    "num_gt_yes",
    "num_gt_no",
    "num_parsed_yes",
    "num_parsed_no",
    "num_parsed_none",
    "num_correct",
    "num_hard_hallucination",
    "num_false_negative_error",
    "num_primary_population",
    "hallucination_rate_in_primary_population",
)

OBJECT_NAME_AUDIT_COLUMNS = (
    "dataset_name",
    "subset",
    "object_name",
    "num_records",
    "num_correct",
    "num_hallucination",
)

SAMPLE_OVERLAP_AUDIT_COLUMNS = (
    "overlap_key",
    "left_dataset",
    "right_dataset",
    "left_subset",
    "right_subset",
    "overlap_count",
)
