"""用过去完整姿态估计当前安装姿态及窗口平均运动率，不使用未来测量。"""

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation


class OrientationRate:
    def __init__(self, window_s=.1):
        self.window_ns = int(window_s*1e9)
        self.history = deque()
        self.variance = np.full(3, .009**2)

    def observe(self, stamp_ns, quaternion_xyzw):
        if self.history and stamp_ns <= self.history[-1][0]:
            raise ValueError("IMU_STAMP_NOT_INCREASING")
        self.history.append((int(stamp_ns), tuple(quaternion_xyzw)))
        while len(self.history) > 2 and stamp_ns-self.history[1][0] >= self.window_ns:
            self.history.popleft()
        if len(self.history) < 3:
            return None
        stamps = np.asarray([row[0]-stamp_ns for row in self.history], dtype=float)*1e-9
        if np.ptp(stamps) < .02:
            return None
        rotations = Rotation.from_quat([row[1] for row in self.history])
        # 旋转对数在当前完整船体坐标中拟合，跨 +/-pi 不会产生假角速度。
        vectors = (rotations[-1].inv()*rotations).as_rotvec()
        centered = stamps-stamps.mean()
        denominator = float(centered@centered)
        slope = centered@(vectors-vectors.mean(axis=0))/denominator
        residual = vectors-vectors.mean(axis=0)-centered[:, None]*slope
        self.variance = np.sum(residual**2, axis=0)/max(1, len(stamps)-2)/denominator
        return tuple(float(value) for value in slope)

    def pose_at(self, stamp_ns, quaternion_xyzw):
        """二次拟合保留转动加速度，返回姿态及角度残差协方差。"""
        rows = [row for row in self.history if row[0] <= stamp_ns]
        if len(rows) < 6:
            return None
        times = np.asarray([row[0]-stamp_ns for row in rows], dtype=float)*1e-9
        reference = Rotation.from_quat(quaternion_xyzw)
        vectors = (reference.inv()*Rotation.from_quat([row[1] for row in rows])).as_rotvec()
        design = np.column_stack((np.ones(len(times)), times, times**2))
        coefficients, _, rank, _ = np.linalg.lstsq(design, vectors, rcond=None)
        if rank != 3:
            return None
        fitted = reference*Rotation.from_rotvec(coefficients[0])
        # 使用原姿态残差保守表达杆臂误差，不能混用单位不同的运动率方差。
        residual = vectors-design@coefficients
        covariance = residual.T@residual/(len(rows)-3)
        return tuple(float(value) for value in fitted.as_quat()), covariance
