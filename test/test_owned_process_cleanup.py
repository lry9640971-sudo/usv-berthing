"""回归父进程退出后仍有子进程存活的真实失败方式。"""

from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from owned_processes import process_group_members, terminate_process_group


def test_cleanup_reaps_group_after_leader_exits():
    code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])"
    leader = subprocess.Popen([sys.executable, "-c", code], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        leader.wait(timeout=3.)
        assert process_group_members(leader.pid)
        assert terminate_process_group(leader.pid, graceful_s=.2)==[]
        assert process_group_members(leader.pid)==[]
    finally:
        terminate_process_group(leader.pid, graceful_s=.1)
