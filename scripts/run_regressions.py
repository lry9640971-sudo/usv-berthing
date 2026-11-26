"""运行唯一源码功能回归并保存结果；不启动生产节点。"""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "latest"
    out = ROOT / "evidence" / ("regression_" + label)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    packages = [ROOT / "control_ws/src/competition_user"]
    packages += [ROOT / "perception_ws/src" / name for name in (
        "phase16z_s3_runtime", "phase17_berthing_perception", "phase17_surround_bev")]
    env["PYTHONPATH"] = ":".join(map(str, packages)) + ":" + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    tests = sorted((ROOT / "control_ws/src/competition_user/test").glob("test_*.py"))
    tests = [path for path in tests if path.name not in {"test_copyright.py", "test_flake8.py", "test_pep257.py"}]
    tests += [path for package in ("phase16z_s3_runtime", "phase17_berthing_perception", "phase17_surround_bev")
              for path in sorted((ROOT / "perception_ws/src" / package / "test").glob("test_*.py"))
              if path.name not in {"test_copyright.py", "test_flake8.py", "test_pep257.py"}]
    tests += sorted((ROOT / "test").glob("test_*.py"))
    command = [sys.executable, "-m", "pytest", *map(str, tests), "-q", "-p", "no:cacheprovider"]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=180)
    output = result.stdout + result.stderr
    (out / "pytest.log").write_text(output, encoding="utf-8")
    syntax = []
    for folder in (ROOT / "control_ws/src", ROOT / "perception_ws/src"):
        for path in folder.rglob("*.py"):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeError) as error:
                syntax.append({"path": str(path), "error": str(error)})
    (out / "summary.json").write_text(json.dumps({"exit_code": result.returncode, "syntax_errors": syntax,
                                                "source_root": str(ROOT), "command": command}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output[-14000:])
    print(json.dumps({"syntax_errors": syntax, "evidence": str(out)}, ensure_ascii=False))
    raise SystemExit(result.returncode or bool(syntax))


if __name__ == "__main__":
    main()
