"""管理明确创建的进程组，父进程退出不能被误认为全部子进程退出。"""

import os
import signal
import subprocess
import time


def process_group_members(group_id):
    rows = subprocess.run(["ps", "-eo", "pid,pgid,stat,args"], capture_output=True,
        text=True, encoding="utf-8", check=True).stdout.splitlines()
    found = []
    for row in rows:
        values = row.strip().split(None, 3)
        if len(values)!=4 or not values[0].isdigit() or not values[1].isdigit():
            continue
        if int(values[1])==group_id and not values[2].startswith("Z"):
            found.append({"pid": int(values[0]), "args": values[3]})
    return found


def terminate_process_group(group_id, graceful_s=6.):
    """只清理指定且经记录的组；即使组长已退出，仍检查和回收其存活子进程。"""
    for action, timeout in ((signal.SIGINT, graceful_s), (signal.SIGTERM, 3.), (signal.SIGKILL, 2.)):
        if not process_group_members(group_id):
            return []
        try:
            os.killpg(group_id, action)
        except ProcessLookupError:
            return []
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            if not process_group_members(group_id):
                return []
            time.sleep(.1)
    return process_group_members(group_id)
