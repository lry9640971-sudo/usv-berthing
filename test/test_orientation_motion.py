"""姿态运动估计保持真实快速转动，同时消除零限位安装关节的高频分量。"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from phase16z_s3_runtime.s3_orientation_rate import OrientationRate


@pytest.mark.parametrize('rate', [.4, 3.6, -3.6])
@pytest.mark.parametrize('window', [.1, .2])
def test_orientation_motion_preserves_actual_fast_turns_and_yaw_wrap(rate, window):
    estimator = OrientationRate(window)
    for step in range(150):
        t = step*.008
        rotation = Rotation.from_euler('z', 3.+rate*t)
        result = estimator.observe(int((1.+t)*1e9), rotation.as_quat())
    assert result == pytest.approx((0., 0., rate), abs=1e-10)


def test_orientation_motion_removes_mount_jitter_without_angular_clipping():
    estimator = OrientationRate()
    rates = []
    for step in range(250):
        t = step*.008
        # 安装关节角度很小，瞬时速度仍可达数 rad/s；直接拿去修正 GPS 会放大假侧移。
        yaw = .2*t+.006*np.sin(2*np.pi*50*t)
        result = estimator.observe(int((1.+t)*1e9), Rotation.from_euler('z', yaw).as_quat())
        if step > 30:
            rates.append(result[2])
    assert np.percentile(np.abs(np.asarray(rates)-.2), 95) < .025


def test_three_dimensional_motion_uses_body_rotation_instead_of_euler_yaw_only():
    estimator = OrientationRate()
    initial = Rotation.from_euler('xyz', [.5, -.4, 2.9])
    omega = np.asarray([.1, -.2, .3])
    for step in range(80):
        t = step*.008
        rotation = initial*Rotation.from_rotvec(omega*t)
        result = estimator.observe(int((1.+t)*1e9), rotation.as_quat())
    assert result == pytest.approx(omega, abs=1e-10)
