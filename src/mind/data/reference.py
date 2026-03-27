"""Reference split utilities."""

from __future__ import annotations


def build_reference_candidates(
    coco_instances: dict[str, list[dict[str, object]]],
    *,
    allowed_objects: set[str],
    exclude_image_ids: set[int],
) -> list[dict[str, object]]:
    image_names = {
        int(image["id"]): str(image["file_name"])
        for image in coco_instances.get("images", [])
    }
    category_names = {
        int(category["id"]): str(category["name"])
        for category in coco_instances.get("categories", [])
    }
    grouped: dict[int, set[str]] = {}
    for annotation in coco_instances.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id in exclude_image_ids:
            continue
        category_name = category_names.get(int(annotation["category_id"]))
        if category_name not in allowed_objects:
            continue
        grouped.setdefault(image_id, set()).add(category_name)

    candidates = []
    for image_id in sorted(grouped):
        candidates.append(
            {
                "image_id": image_id,
                "file_name": image_names[image_id],
                "object_names": sorted(grouped[image_id]),
            }
        )
    return candidates
