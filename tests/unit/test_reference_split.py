from __future__ import annotations

from mind.data import build_reference_candidates


def test_build_reference_candidates_filters_categories_and_excluded_images() -> None:
    coco_instances = {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg"},
            {"id": 2, "file_name": "000000000002.jpg"},
            {"id": 3, "file_name": "000000000003.jpg"},
        ],
        "categories": [
            {"id": 1, "name": "dog"},
            {"id": 2, "name": "cat"},
            {"id": 3, "name": "bus"},
        ],
        "annotations": [
            {"id": 11, "image_id": 1, "category_id": 1},
            {"id": 12, "image_id": 1, "category_id": 2},
            {"id": 13, "image_id": 2, "category_id": 1},
            {"id": 14, "image_id": 3, "category_id": 3},
        ],
    }

    candidates = build_reference_candidates(
        coco_instances,
        allowed_objects={"dog", "bus"},
        exclude_image_ids={2},
    )

    assert candidates == [
        {
            "image_id": 1,
            "file_name": "000000000001.jpg",
            "object_names": ["dog"],
        },
        {
            "image_id": 3,
            "file_name": "000000000003.jpg",
            "object_names": ["bus"],
        },
    ]


def test_build_reference_candidates_can_cap_images_per_object() -> None:
    coco_instances = {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg"},
            {"id": 2, "file_name": "000000000002.jpg"},
            {"id": 3, "file_name": "000000000003.jpg"},
        ],
        "categories": [{"id": 1, "name": "dog"}],
        "annotations": [
            {"id": 11, "image_id": 1, "category_id": 1},
            {"id": 12, "image_id": 2, "category_id": 1},
            {"id": 13, "image_id": 3, "category_id": 1},
        ],
    }

    candidates = build_reference_candidates(
        coco_instances,
        allowed_objects={"dog"},
        exclude_image_ids=set(),
        max_images_per_object=2,
    )

    assert candidates == [
        {"image_id": 1, "file_name": "000000000001.jpg", "object_names": ["dog"]},
        {"image_id": 2, "file_name": "000000000002.jpg", "object_names": ["dog"]},
    ]
