from __future__ import annotations

from typing import Any, Iterable

QUERY_PARAM_CANDIDATES = ("LLM", "llm", "llm_model", "model", "model_id")


def _read_query_params(request: Any) -> dict[str, str]:
    if request is None:
        return {}

    direct_query_params = getattr(request, "query_params", None)
    if direct_query_params is not None:
        return _to_str_dict(direct_query_params)

    nested_request = getattr(request, "request", None)
    nested_query_params = getattr(nested_request, "query_params", None)
    if nested_query_params is not None:
        return _to_str_dict(nested_query_params)

    return {}


def _to_str_dict(query_params: Any) -> dict[str, str]:
    if query_params is None:
        return {}

    if hasattr(query_params, "multi_items"):
        items = list(query_params.multi_items())
    elif hasattr(query_params, "items"):
        items = list(query_params.items())
    elif isinstance(query_params, dict):
        items = list(query_params.items())
    else:
        return {}

    out: dict[str, str] = {}
    for key, value in items:
        if key is None or value is None:
            continue
        out[str(key)] = str(value)
    return out


def requested_llm_model_from_request(request: Any) -> str | None:
    query_params = _read_query_params(request)
    for key in QUERY_PARAM_CANDIDATES:
        value = query_params.get(key)
        if value is None:
            continue
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def resolve_llm_model(
    explicit_model: str | None,
    request: Any = None,
    *,
    default_model: str,
) -> str:
    if isinstance(explicit_model, str) and explicit_model.strip():
        return explicit_model.strip()

    requested_model = requested_llm_model_from_request(request)
    if requested_model:
        return requested_model

    return default_model


def build_model_choices(
    current_choices: Iterable[tuple[str, str]],
    requested_model: str | None,
) -> list[tuple[str, str]]:
    choices = list(current_choices)
    if not requested_model:
        return choices

    known_values = {value for _, value in choices}
    if requested_model in known_values:
        return choices

    return [(requested_model, requested_model), *choices]