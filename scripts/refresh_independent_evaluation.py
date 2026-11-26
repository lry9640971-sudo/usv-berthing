"""用原独立记录更新完成时速度口径，保存旧结果，不重跑或改动生产。"""

import argparse
import json
from pathlib import Path

from official_evaluation import completion_snapshot, evaluate


def refresh(case):
    def read(name):
        return json.loads((case/name).read_text(encoding='utf-8'))
    previous = read('result.json')
    if previous['mode'] != 'acceptance':
        raise ValueError('正式复算不接受诊断起点运行')
    backup = case/'result_legacy_average_v1.json'
    if not backup.exists():
        backup.write_text(json.dumps(previous, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report, monitor = read('controller/completion.json'), read('monitor.json')
    stamp = report['stamp_sim_s']
    truth = []
    with (case/'sensor_motion_history.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            if 'independent_truth' not in line:
                continue
            row = json.loads(line)
            if row['stream'] == 'independent_truth' and abs(row['stamp']-stamp) <= .30:
                truth.append((row['stamp'], *row['xyz'][:2]))
    snapshot = completion_snapshot(truth, stamp)
    if snapshot is None:
        raise ValueError('原记录缺少完成时刻附近的独立真值，不能借用时间平均判通过')
    monitor['completion_truth'] = snapshot
    result = evaluate(read('scene/manifest.json'), monitor, report, read('controller/runtime_status.json'),
                      previous['elapsed_wall_s'], previous['stop_reason'])
    result.update(mode=previous['mode'], process_return_codes=previous['process_return_codes'],
                  evaluation_revision='instantaneous_completion_v2', source_case=case.name,
                  previous_evaluation='result_legacy_average_v1.json')
    (case/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (case/'completion_truth_instantaneous_v2.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=Path)
    args = parser.parse_args()
    result = refresh(args.case)
    print(json.dumps({key: result[key] for key in ('source_case', 'official_single_run_passed',
        'independent_completion_speed_mps', 'legacy_preceding_0p2s_speed_mps', 'gates')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
