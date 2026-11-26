"""每路最多保留一帧并公平轮转，防止慢处理让同一路长期挤掉其他雷达。"""

from collections import deque
from threading import Condition


class FairLatestCloudMailbox:
    def __init__(self, topics):
        self.topics = frozenset(topics)
        self.pending = {}
        self.order = deque()
        self.condition = Condition()
        self.unfinished = 0
        self.closed = False
        self.replaced_by_topic = {topic: 0 for topic in topics}

    def put(self, topic, message):
        if topic not in self.topics:
            raise ValueError("UNKNOWN_CLOUD_TOPIC")
        with self.condition:
            if self.closed:
                return False
            replaced = topic in self.pending
            if replaced:
                self.replaced_by_topic[topic] += 1
            else:
                self.order.append(topic)
                self.unfinished += 1
            self.pending[topic] = message
            self.condition.notify()
            return replaced

    def get(self):
        with self.condition:
            self.condition.wait_for(lambda: bool(self.order) or self.closed)
            if not self.order:
                return None
            topic = self.order.popleft()
            return self.pending.pop(topic), topic

    def task_done(self):
        with self.condition:
            self.unfinished -= 1
            if self.unfinished < 0:
                raise ValueError("CLOUD_QUEUE_TASK_DONE_MISMATCH")
            self.condition.notify_all()

    def qsize(self):
        with self.condition:
            return len(self.pending)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def join(self):
        with self.condition:
            self.condition.wait_for(lambda: self.unfinished == 0)
