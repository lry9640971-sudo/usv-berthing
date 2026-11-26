"""任务与预测共享的位置反馈；真实侧移进入目标，不假定船能横向平移。"""

import math

from .dynamics import wrap


def point_targets(state, goal_xy, maximum_speed=.6):
    dx, dy = goal_xy[0]-state.x, goal_xy[1]-state.y
    vx, vy = .4*dx, .4*dy
    norm = math.hypot(vx, vy)
    scale = min(1., maximum_speed/max(norm, 1e-12))
    vx, vy = vx*scale+math.sin(state.yaw)*state.sway, vy*scale-math.cos(state.yaw)*state.sway
    bearing = math.atan2(vy, vx)
    error = wrap(bearing-state.yaw)
    sign = 1. if abs(error) <= math.pi/2 else -1.
    wanted = bearing if sign > 0 else wrap(bearing+math.pi)
    error = wrap(wanted-state.yaw)
    speed = sign*min(maximum_speed, math.hypot(vx, vy))*max(0., math.cos(error))
    return speed, max(-.5, min(.5, .8*error))


def terminal_surge(state, goal_xy, reference_yaw):
    dx, dy = goal_xy[0]-state.x, goal_xy[1]-state.y
    along = math.cos(reference_yaw)*dx+math.sin(reference_yaw)*dy
    # 进场允许.75m/s，剩余距离缩小时减速；实际刹停和转向仍由同一受力模型优化。
    surge = max(-.2, min(.75, .4*along))
    if math.hypot(dx, dy) < 1.2:
        surge = max(-.18, min(.18, surge))
    return surge
