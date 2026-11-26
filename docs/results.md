# Results and reproduction

The acceptance records come from the final delivery at `3b6341b` with autonomy frozen at `8602b06`. The [matrix summary](../evidence/official_matrix/matrix_progress.json) indexes five runs from the original competition generator at `75babf0`. The runs used Gazebo Harmonic with headless EGL and Mesa 25.2.8 llvmpipe with six rendering threads. The recorded numerical libraries were NumPy 1.26.4, SciPy 1.11.4, and Shapely 2.0.3.

- **Position error** is horizontal distance from the independent vessel pose at completion to the selected berth's reference target.
- **Completion speed** is the magnitude of the derivative at completion from a quadratic fit to horizontal truth positions. The fit uses samples within ±40 ms and requires observations on both sides of the report timestamp. The two earliest runs were rescored from their original observations after this evaluation change.
- **Time and collisions** come from the supervisor and the Gazebo contact stream. Wall time is measured from controller launch, and the cycle budget covers planning and the full control cycle. The evaluator recorded `NOT_OBSERVED` for berth contact on every run.

[`evidence/runs`](../evidence/runs) contains the saved result, controller report, monitor summary, scenario manifest, and short completion-time truth trace for each seed. The original numeric fields and samples are retained. Machine-specific paths in the manifests and the regression metadata were replaced with relative references. The simulator regenerates the worlds from the seed and external platform.

To recompute the five saved runs, use the evaluation script on a temporary copy. This uses the saved run times and contact evidence and recomputes the completion measurements in place within that copy. It requires Python and NumPy and starts no ROS 2 nodes.

```bash
check_dir=$(mktemp -d)
cp -R evidence/runs "$check_dir/runs"
for seed in 20261003 20261004 20261005 20261006 20261007; do
  python3 scripts/refresh_independent_evaluation.py "$check_dir/runs/seed$seed"
done
```

The [236-test runtime record](../evidence/regression_navigation_attitude_projection_01/summary.json) and [238-test final evaluation record](../evidence/regression_instantaneous_completion_contract_01/summary.json) refer to the same autonomy revision. Their command lists describe the original delivery, including legacy controller tests omitted from this release. The final suite added explicit checks for instantaneous completion velocity. The preserved fixture includes a case whose speed computed from the independent samples exceeded the limit even though the controller's report was below it. See [`test_independent_evaluation.py`](../test/test_independent_evaluation.py).
