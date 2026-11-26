"""靠泊感知的多路输入新鲜度选择。"""

from __future__ import annotations

from collections.abc import Mapping


def select_fresh_inputs(
    stamps_by_name: Mapping[str, int],
    *,
    minimum_inputs: int,
    maximum_age_ns: int,
) -> tuple[tuple[str, ...], int, int] | None:
    """选择相对最新时间仍新鲜的输入，返回名称、锚点和时间跨度。"""

    minimum = int(minimum_inputs)
    maximum_age = int(maximum_age_ns)
    if minimum < 1 or maximum_age <= 0:
        raise ValueError("FRESH_INPUT_PARAMETER_INVALID")
    valid = {str(name): int(stamp) for name, stamp in stamps_by_name.items() if int(stamp) > 0}
    if not valid:
        return None
    anchor = max(valid.values())
    selected = tuple(
        sorted(name for name, stamp in valid.items() if 0 <= anchor - stamp <= maximum_age)
    )
    if len(selected) < minimum:
        return None
    span = anchor - min(valid[name] for name in selected)
    return selected, anchor, span


__all__ = ["select_fresh_inputs"]
