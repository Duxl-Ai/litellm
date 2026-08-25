from litellm.router_utils.fallback_event_handlers import (
    _check_non_standard_fallback_format,
)


def test_mixed_direct_fallback_targets_are_non_standard_format() -> None:
    fallbacks = [
        "backup-a",
        {"model": "backup-b", "temperature": 0},
    ]

    assert _check_non_standard_fallback_format(fallbacks) is True


def test_model_group_mapping_remains_standard_format() -> None:
    fallbacks = [{"primary": ["backup-a", {"model": "backup-b"}]}]

    assert _check_non_standard_fallback_format(fallbacks) is False


def test_list_valued_model_direct_fallback_remains_non_standard_format() -> None:
    fallbacks = [{"model": ["backup-a", "backup-b"], "temperature": 0}]

    assert _check_non_standard_fallback_format(fallbacks) is True
