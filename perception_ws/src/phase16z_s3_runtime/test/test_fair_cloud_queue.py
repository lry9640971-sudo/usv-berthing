"""独立检查多路覆盖与替换规则，不仅验证总队列长度。"""

from phase16z_s3_runtime.s3_fair_cloud_queue import FairLatestCloudMailbox


def test_fast_sensor_cannot_evict_other_pending_sensors():
    queue = FairLatestCloudMailbox(("front", "left", "right"))
    queue.put("front", 1)
    queue.put("left", 1)
    queue.put("right", 1)
    for value in range(2, 100):
        queue.put("front", value)
    seen = []
    for _ in range(3):
        seen.append(queue.get())
        queue.task_done()
    assert seen == [(99, "front"), (1, "left"), (1, "right")]
    assert queue.replaced_by_topic == {"front": 98, "left": 0, "right": 0}
    queue.close()
    queue.join()
    assert queue.get() is None


def test_arriving_during_processing_goes_after_other_sensors():
    queue = FairLatestCloudMailbox(("a", "b"))
    queue.put("a", 1)
    queue.put("b", 1)
    assert queue.get() == (1, "a")
    queue.put("a", 2)
    queue.task_done()
    assert queue.get() == (1, "b")
    queue.task_done()
    assert queue.get() == (2, "a")
    queue.task_done()
    queue.close()
    queue.join()
