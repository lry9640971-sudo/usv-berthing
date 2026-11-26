"""明确任务状态和泊位生命周期；没有由推力发布反改的隐藏状态。"""

from dataclasses import dataclass
from enum import Enum
import math

from .dynamics import wrap
from .geometry import transform_xy
from .planner import Guidance
from .guidance import terminal_surge


class MissionState(str, Enum):
    NAVIGATE = "NAVIGATE"
    APPROACH = "APPROACH"
    FINAL = "FINAL"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class Bay:
    bay_id: int
    entry_pose: tuple
    final_pose: tuple

    @classmethod
    def from_payload(cls, row):
        def pose(name):
            values = tuple(float(row[name][key]) for key in ("x", "y", "yaw"))
            if not all(math.isfinite(v) for v in values):
                raise ValueError("泊位姿态非有限值")
            return values
        return cls(int(row["bay_id"]), pose("entry_pose"), pose("final_pose"))


class Mission:
    def __init__(self):
        self.state = MissionState.NAVIGATE
        self.bay = None
        self.bay_stamp = -math.inf
        self.dock_prior = None
        self.dock_geometry_pose = None
        self.geometry_stamp = -math.inf
        self.free_bays = []
        self.occupied_measurements = {}
        self.completed_since = None
        self.reason = "WAIT_DOCK_PRIOR"

    def accept_bays(self, payload, now):
        if payload["frame_id"] != "team_enu":
            raise ValueError("泊位坐标不是 ENU")
        stamp = int(payload["stamp_ns"]) / 1e9
        if stamp <= self.geometry_stamp or not -.05 <= now - stamp <= 3.:
            return
        rows = payload["candidates"]
        if payload["status"] == "READY":
            self.dock_geometry_pose = tuple(payload["dock_pose_xyyaw_m_rad"])
            if len(self.dock_geometry_pose) != 3 or not all(math.isfinite(v) for v in self.dock_geometry_pose):
                raise ValueError("确认码头的几何姿态无效")
            self.geometry_stamp = stamp
        occupied = set()
        for row in rows:
            bay_id = int(row["bay_id"])
            previous_stamp, count = self.occupied_measurements.get(bay_id, (-math.inf, 0))
            valid = payload["status"] == "READY" and row["valid"]
            if valid and row["occupancy_status"] == "FREE":
                self.occupied_measurements.pop(bay_id, None)
            elif valid and row["occupancy_status"] == "OCCUPIED":
                # 一个噪声帧不能改选目标；当前物理避碰仍由同一完整轨迹检查负责。
                count = min(2, count + 1) if stamp - previous_stamp <= 3. else 1
                self.occupied_measurements[bay_id] = (stamp, count)
                if count == 2:
                    occupied.add(bay_id)
            elif bay_id in self.occupied_measurements:
                # 未知打断连续确认，但不能把已有占用疑虑改写为空闲。
                self.occupied_measurements[bay_id] = (stamp, 0)
        self.free_bays = ([Bay.from_payload(row) for row in rows if row["valid"] and row["occupancy_status"] == "FREE"]
                          if payload["status"] == "READY" else [])
        if self.bay is not None:
            if self.bay.bay_id in occupied:
                self.revoke("BAY_OCCUPIED")
            else:
                updated = next((bay for bay in self.free_bays if bay.bay_id == self.bay.bay_id), None)
                if updated is not None:
                    self.bay, self.bay_stamp = updated, stamp

    def revoke(self, reason):
        self.bay = None
        self.completed_since = None
        self.state, self.reason = MissionState.NAVIGATE, reason

    @property
    def pending_occupancy(self):
        return self.bay is not None and self.bay.bay_id in self.occupied_measurements

    def update(self, vessel, now, can_complete=True):
        if self.state in (MissionState.COMPLETE, MissionState.FAILED):
            return
        if self.bay is not None and now - self.bay_stamp > 3.:
            self.revoke("BAY_CONFIRMATION_EXPIRED")
        if self.bay is None and self.free_bays and now - self.geometry_stamp <= 3.:
            self.bay = min(self.free_bays, key=lambda bay: math.hypot(vessel.x-bay.entry_pose[0], vessel.y-bay.entry_pose[1]))
            self.bay_stamp = self.geometry_stamp
            self.state, self.reason = MissionState.APPROACH, "OBSERVED_FREE_BAY"
        if self.bay is None:
            return
        if self.state == MissionState.APPROACH:
            entry = self.bay.entry_pose
            if math.hypot(vessel.x-entry[0], vessel.y-entry[1]) < 1. and abs(wrap(vessel.yaw-entry[2])) < .30:
                self.state, self.reason = MissionState.FINAL, "ENTRY_REACHED"
        if can_complete is not None:
            self.check_completion(vessel, now, can_complete)

    def check_completion(self, vessel, now, can_complete):
        # 生产节点在本次规划完成后联合确认，不能先发完成再发现无安全轨迹。
        if self.state != MissionState.FINAL:
            return
        final = self.bay.final_pose
        distance = math.hypot(vessel.x-final[0], vessel.y-final[1])
        # 官方仿真3.2.2按完成时的位置与船速评分，没有额外角速度门。
        # 内部位置留5厘米测量余量；按完成标志时刻判断，不增加官方未要求的驻留门。
        if can_complete and not self.pending_occupancy and distance <= .15 and vessel.speed <= .20:
            self.state, self.reason = MissionState.COMPLETE, "POSITION_AND_SPEED_REACHED"
        else:
            self.completed_since = None

    def guidance(self, vessel):
        if self.bay is None:
            if self.dock_prior is None:
                return None
            # 公共码头只引导航行至入口外；不能充当已识别的停靠终点。
            goal = tuple(transform_xy([(-12., 3.)], self.dock_prior)[0])
            distance = math.hypot(goal[0]-vessel.x, goal[1]-vessel.y)
            return self._point_guidance(vessel, goal, .9 if distance > 20. else .5)
        if self.state == MissionState.APPROACH:
            goal = self.bay.entry_pose
            distance = math.hypot(goal[0]-vessel.x, goal[1]-vessel.y)
            guidance = self._point_guidance(vessel, goal[:2], min(.6, .3 * distance), goal[2])
            if distance < 1.:
                return Guidance(guidance.goal_xy, 0., max(-.5, min(.5, .8*wrap(goal[2]-vessel.yaw))),
                                goal[2], use_route=False, position_tolerance_m=1.)
            return guidance
        final = self.bay.final_pose
        dx, dy = final[0]-vessel.x, final[1]-vessel.y
        c, s = math.cos(final[2]), math.sin(final[2])
        along, cross = c*dx+s*dy, -s*dx+c*dy
        # 最终进场由同一受力优化器处理横偏、制动及必要的倒车；目标不再跳回入口。
        distance = math.hypot(dx, dy)
        surge = terminal_surge(vessel, final[:2], final[2])
        # 实测侧向速度进入引导，避免仅向终点瞄准却持续被风推离中心线。
        cross_drift = vessel.sway*math.cos(vessel.yaw-final[2])
        wanted_yaw = final[2] + math.atan2(.4*cross-cross_drift, max(.12, abs(surge))) * (1. if surge >= 0 else -1.)
        yaw_error = wrap(wanted_yaw-vessel.yaw)
        surge *= max(0., math.cos(yaw_error))
        return Guidance(final[:2], surge, max(-.3, min(.3, .8*yaw_error)), final[2], distance < 2., feedback="terminal")

    @staticmethod
    def _point_guidance(vessel, goal, speed, goal_yaw=None):
        desired = math.atan2(goal[1]-vessel.y, goal[0]-vessel.x) + (math.pi if speed < 0 else 0.)
        error = wrap(desired-vessel.yaw)
        surge = speed * max(0., math.cos(error)) ** 2
        return Guidance(tuple(goal), surge, max(-.5, min(.5, .8*error)), goal_yaw)
