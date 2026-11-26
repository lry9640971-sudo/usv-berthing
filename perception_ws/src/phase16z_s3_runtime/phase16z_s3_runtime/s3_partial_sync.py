"""为频率不一致的多路点云选择单调、有限等待的时间组。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def pop_ready_group(
    rows_by_topic: Mapping[str, Any],
    *,
    tolerance_ns: int,
    last_emitted_stamp_ns: int = 0,
    force: bool = False,
) -> tuple[tuple[Any, ...], int, int] | None:
    """弹出一个时间组，返回行、组时间戳和被清理的过期行数。

    正常运行时优先等待全部输入；一旦后续点云已越过当前窗口，就用窗口内
    已到达的输入形成部分组。这样慢路或断路不会阻塞其余传感器，同时保持
    输出时间严格递增。``force`` 仅用于停机前排空。
    """

    tolerance = int(tolerance_ns)
    if tolerance <= 0:
        raise ValueError("PARTIAL_SYNC_TOLERANCE_INVALID")

    def take_first(queue: Any) -> Any:
        return queue.popleft() if hasattr(queue, "popleft") else queue.pop(0)

    stale = 0
    for queue in rows_by_topic.values():
        while queue and int(queue[0].stamp_ns) <= int(last_emitted_stamp_ns):
            take_first(queue)
            stale += 1

    nonempty = [queue for queue in rows_by_topic.values() if queue]
    if not nonempty:
        return None

    anchor = min(int(queue[0].stamp_ns) for queue in nonempty)
    selected: dict[str, int] = {}
    for topic, queue in rows_by_topic.items():
        choices = [
            (abs(int(row.stamp_ns) - anchor), index)
            for index, row in enumerate(queue)
            if abs(int(row.stamp_ns) - anchor) <= tolerance
        ]
        if choices:
            selected[topic] = min(choices)[1]

    newest = max(int(queue[-1].stamp_ns) for queue in nonempty)
    complete = len(selected) == len(rows_by_topic)
    if not force and not complete and newest <= anchor + tolerance:
        return None

    rows: list[Any] = []
    for topic, index in selected.items():
        queue = rows_by_topic[topic]
        for _ in range(index):
            take_first(queue)
            stale += 1
        rows.append(take_first(queue))
    if not rows:
        return None
    rows.sort(key=lambda row: (int(row.stamp_ns), str(row.topic)))
    primary = max(int(row.stamp_ns) for row in rows)
    return tuple(rows), primary, stale


__all__ = ["pop_ready_group"]
