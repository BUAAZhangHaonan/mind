"""Reference split utilities."""

from __future__ import annotations


def build_reference_candidates(
    coco_instances: dict[str, list[dict[str, object]]],
    *,
    allowed_objects: set[str],
    exclude_image_ids: set[int],
    max_images_per_object: int = 0,
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

    object_counts = {object_name: 0 for object_name in allowed_objects}
    candidates = []
    for image_id in sorted(grouped):
        selected_objects = []
        for object_name in sorted(grouped[image_id]):
            if max_images_per_object > 0 and object_counts.get(object_name, 0) >= max_images_per_object:
                continue
            selected_objects.append(object_name)
        if not selected_objects:
            continue
        candidates.append(
            {
                "image_id": image_id,
                "file_name": image_names[image_id],
                "object_names": selected_objects,
            }
        )
        for object_name in selected_objects:
            object_counts[object_name] = object_counts.get(object_name, 0) + 1
    return candidates
