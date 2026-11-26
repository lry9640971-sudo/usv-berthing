"""只读记录本轮进程内存及交换压力，不改变仿真或控制决策。"""

import json
import os
from pathlib import Path
import signal
import sys
import time


def read_values(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.replace(":", "").split()
        if len(parts) > 1 and parts[1].isdigit():
            result[parts[0]] = int(parts[1])
    return result


def main():
    out, parent = Path(sys.argv[1]), int(sys.argv[2])
    started, active = time.monotonic(), True

    def stop(*_):
        nonlocal active
        active = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with (out / "resource_history.jsonl").open("a", encoding="utf-8") as stream:
        while active and Path(f"/proc/{parent}").exists():
            registry = json.loads((out / "owned_processes.json").read_text(encoding="utf-8"))
            groups = {item["process_group"]: name for name, item in registry.items()}
            rows = []
            for folder in Path("/proc").iterdir():
                if not folder.name.isdigit():
                    continue
                try:
                    stat = (folder / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
                    group = int(stat[2])
                    if group not in groups or stat[0] == "Z":
                        continue
                    values = read_values(folder / "status")
                    command = (folder / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8")
                    if groups[group] == "simulation" and "gz sim" in command:
                        maps = (folder / "maps").read_text(encoding="utf-8")
                        if "libgallium" in maps and not (out / "graphics_runtime_maps.txt").exists():
                            (out / "graphics_runtime_maps.txt").write_text(maps, encoding="utf-8")
                except (FileNotFoundError, ProcessLookupError):
                    # 进程在只读采样期间正常退出，不把退出伪造成内存为零。
                    continue
                rows.append({"pid": int(folder.name), "group": groups[group], "command": command,
                             "rss_kb": values.get("VmRSS"), "anonymous_kb": values.get("RssAnon"),
                             "swap_kb": values.get("VmSwap"), "threads": values.get("Threads"),
                             "major_faults": int(stat[9]), "cpu_ticks": int(stat[11]) + int(stat[12])})
            memory = read_values(Path("/proc/meminfo"))
            vm = read_values(Path("/proc/vmstat"))
            payload = {"elapsed_wall_s": time.monotonic() - started, "wall_unix_s": time.time(),
                       "mem_available_kb": memory["MemAvailable"], "swap_free_kb": memory["SwapFree"],
                       "swap_in_pages": vm["pswpin"], "swap_out_pages": vm["pswpout"],
                       "memory_pressure": Path("/proc/pressure/memory").read_text(encoding="utf-8").strip(),
                       "processes": rows}
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stream.flush()
            temporary = out / "resource_status.tmp"
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(out / "resource_status.json")
            time.sleep(2.)


if __name__ == "__main__":
    main()
