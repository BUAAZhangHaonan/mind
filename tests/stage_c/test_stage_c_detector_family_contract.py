from __future__ import annotations

from .conftest import stage_c_attr


def test_stage_c_compares_exactly_five_methods_and_marks_logistic_as_comparator() -> None:
    support = stage_c_attr("stage_c_support", "stage_c_method_contract")()

    assert support["methods"] == [
        "single_vmf",
        "mixture_vmf",
        "radius_ball",
        "knn",
        "logistic",
    ]
    assert set(support["support_methods"]) == {"single_vmf", "mixture_vmf", "radius_ball", "knn"}
    assert support["comparator_methods"] == ["logistic"]
    assert support["method_roles"]["logistic"] == "supervised_comparator"
    assert all(support["method_roles"][name] == "support_estimator" for name in support["support_methods"])
