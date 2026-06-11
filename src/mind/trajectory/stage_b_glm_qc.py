"""Offline GLM answer parsing and quality-control helpers for Stage B."""

from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from mind.models.registry import REQUIRED_MODEL_ALIASES

from .stage_b_manifest import StageBPanelManifest, stream_stage_b_full_cache_entries


GLM_MODEL_ALIAS = "glm-4.6v-flash"
DEFAULT_GLM_QC_DATASETS = ("repope", "pope", "dash-b")
ANSWER_TEXT_FIELDS = ("answer_text", "raw_answer", "answer", "generation", "response", "model_response")

_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)
_NON_PARSEABLE_HINTS = (
    "cannot determine",
    "can't determine",
    "cannot tell",
    "can't tell",
    "not enough information",
    "not clear",
    "unclear",
    "ambiguous",
    "unknown",
    "not sure",
    "unsure",
    "unable to determine",
    "undetermined",
    "indeterminate",
    "i do not know",
    "i don't know",
)


def parse_glm_yes_no_answer(value: object) -> int | None:
    """Parse a GLM answer into 1 for yes, 0 for no, or None when unsafe."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return int(value)

    text = str(value).strip()
    if not text:
        return None
    lowered = " ".join(text.lower().split())
    if any(hint in lowered for hint in _NON_PARSEABLE_HINTS):
        return None

    compact = lowered.strip(" \t\r\n.!,;:\"'`()[]{}")
    if compact in {"yes", "y", "true", "1"}:
        return 1
    if compact in {"no", "n", "false", "0"}:
        return 0

    yes_matches = list(_YES_RE.finditer(lowered))
    no_matches = list(_NO_RE.finditer(lowered))
    if yes_matches and no_matches:
        return None
    if yes_matches:
        first = yes_matches[0]
        if first.start() <= 8 or "answer is yes" in lowered or "the answer is yes" in lowered:
            return 1
    if no_matches:
        first = no_matches[0]
        if first.start() <= 8 or "answer is no" in lowered or "the answer is no" in lowered:
            return 0

    if any(phrase in lowered for phrase in ("not visible", "no visible", "is absent", "not present")):
        return 0
    if any(phrase in lowered for phrase in ("is visible", "are visible", "is present", "there is a")):
        return 1
    if lowered.startswith(("there is no ", "there are no ", "does not ", "do not ")):
        return 0

    return None


def classify_glm_answer_qc(entry: Mapping[str, object]) -> dict[str, object]:
    """Classify one cached GLM row without mutating it."""

    row = dict(entry)
    answer_text = _answer_text(row)
    parsed = parse_glm_yes_no_answer(answer_text)
    parseable = parsed is not None
    reason = (
        "answer_text parsed by offline yes/no parser"
        if parseable
        else "GLM answer_text is not parseable by the offline yes/no parser"
    )
    row.update(
        {
            "answer_text": answer_text,
            "parseable": parseable,
            "derived_parsed_answer": parsed,
            "reason": reason,
        }
    )
    return row


def scan_glm_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    model_alias: str = GLM_MODEL_ALIAS,
) -> list[dict[str, object]]:
    """Classify GLM rows from an iterable of cache entries."""

    results = []
    for row in rows:
        alias = str(row.get("model_alias") or row.get("model_name") or "")
        if alias == model_alias:
            results.append(classify_glm_answer_qc(row))
    return results


def scan_glm_cache_rows(
    manifest: StageBPanelManifest,
    full_cache_root: Path | str,
    *,
    model_alias: str = GLM_MODEL_ALIAS,
    dataset_families: Sequence[str] = DEFAULT_GLM_QC_DATASETS,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """Scan GLM rows from the unified full-cache manifest."""

    model_row = next(
        (row for row in manifest.models if str(row.get("model_alias", "")) == model_alias),
        None,
    )
    if model_row is None:
        raise ValueError(f"GLM panel model not found in manifest: {model_alias}")

    results: list[dict[str, object]] = []
    for family in dataset_families:
        for entry in stream_stage_b_full_cache_entries(
            model_row,
            full_cache_root,
            dataset_family=family,
            include_tensors=False,
        ):
            results.append(classify_glm_answer_qc(entry))
            if limit is not None and len(results) >= int(limit):
                return results
    return results


def summarize_glm_qc(qc_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize GLM QC rows."""

    counts = Counter("parseable" if row.get("parseable") else "nonparseable" for row in qc_rows)
    nonparseable_rows = [dict(row) for row in qc_rows if not row.get("parseable")]
    return {
        "num_rows": len(qc_rows),
        "num_parseable": counts["parseable"],
        "num_nonparseable": counts["nonparseable"],
        "blocked": False,
        "has_nonparseable_rows": bool(nonparseable_rows),
        "nonparseable_rows": nonparseable_rows,
    }


def apply_glm_qc_exclusion(
    *,
    panel_models: Sequence[str] = REQUIRED_MODEL_ALIASES,
    qc_rows: Sequence[Mapping[str, object]],
    model_alias: str = GLM_MODEL_ALIAS,
) -> dict[str, object]:
    """Exclude GLM when QC rows prove its answers cannot be parsed."""

    panel = [str(model) for model in panel_models]
    excluded: dict[str, str] = {}
    if any(
        str(row.get("model_alias") or row.get("model_name") or model_alias) == model_alias
        and row.get("parseable") is False
        for row in qc_rows
    ):
        excluded[model_alias] = "excluded because GLM answer_text is not parseable"

    return {
        "blocked": False,
        "excluded_models": excluded,
        "included_models": [model for model in panel if model not in excluded],
    }


def write_glm_qc_reports(
    qc_rows: Sequence[Mapping[str, object]],
    *,
    json_path: Path | str,
    markdown_path: Path | str,
) -> dict[str, object]:
    """Write JSON and Markdown QC reports."""

    payload = {
        "stage": "stage_b_glm_answer_qc",
        "model_alias": GLM_MODEL_ALIAS,
        "summary": summarize_glm_qc(qc_rows),
        "rows": [dict(row) for row in qc_rows],
        "cache_entries_mutated": False,
    }
    output_json = Path(json_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    output_md = Path(markdown_path)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_glm_qc_markdown(payload), encoding="utf-8")
    return payload


def render_glm_qc_markdown(payload: Mapping[str, object]) -> str:
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    lines = [
        "# Stage B GLM Answer QC",
        "",
        f"- model_alias: {payload.get('model_alias', GLM_MODEL_ALIAS)}",
        f"- rows_checked: {summary.get('num_rows', 0)}",
        f"- parseable: {summary.get('num_parseable', 0)}",
        f"- nonparseable: {summary.get('num_nonparseable', 0)}",
        f"- cache_entries_mutated: {str(payload.get('cache_entries_mutated', False)).lower()}",
        "",
        "## Non-Parseable Rows",
        "",
        "| sample_id | reason | answer_text |",
        "| --- | --- | --- |",
    ]
    nonparseable = summary.get("nonparseable_rows", [])
    if isinstance(nonparseable, Sequence) and not isinstance(nonparseable, (str, bytes)):
        for row in nonparseable:
            if isinstance(row, Mapping):
                lines.append(
                    "| {sample_id} | {reason} | {answer_text} |".format(
                        sample_id=_md_cell(row.get("sample_id", "")),
                        reason=_md_cell(row.get("reason", "")),
                        answer_text=_md_cell(row.get("answer_text", "")),
                    )
                )
    return "\n".join(lines) + "\n"


def _answer_text(row: Mapping[str, object]) -> str:
    for field in ANSWER_TEXT_FIELDS:
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return ""


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()
