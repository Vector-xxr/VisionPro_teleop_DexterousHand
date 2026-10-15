#!/usr/bin/env python3
"""末端执行器这一层的接缝。灵巧手和夹爪共用一个接口,数采和遥操都只认这个接口。

三种实现:

    InspireHandEE   因时 RH56,6 维,0~1000(0 合拢)。走 `inspire_hand6.py`
    GripperEE       两指夹爪,1 维,0~1(0 合拢)。**驱动还没写,见下**
    NullEE          0 维,什么都不驱动

**夹爪现在没接线。** 所以默认用的是 `GripperEE(port=None)` —— 它照样接收指令、
照样把指令录进数据,只是不驱动任何硬件,`connected` 是 False。这样臂这一侧的数据
现在就能采,夹爪那一维记的是操作员的意图;接上线之后补一个 `_write()` 就行,
已经录的数据格式不用变。

要接真夹爪,需要先确认三件事,少一件都写不对:

  1. **型号和通信方式** —— Modbus RTU 走串口 / CAN / 控制器末端 IO 数字量。
     前两种在本机开一个口,第三种要走 xCoreSDK 的 IO 接口,和臂共用连接。
  2. **是位置控制还是开关量** —— 能给 0~1 连续开度,还是只有张/合两个状态。
     只有开关量的话数据里那一维就是 0/1,策略学出来也只能是 0/1。
  3. **有没有位置反馈** —— 有就 `measured()` 读真值,没有就只能拿指令当实测,
     数据里 state 和 action 的夹爪维会完全相同,训练时要知道这一点。
"""
from __future__ import annotations

import numpy as np


class EndEffector:
    """统一接口。dof / lo / hi 决定数据里那几维长什么样。"""

    name = "none"
    dof = 0
    lo = 0.0                 # 合拢端
    hi = 1.0                 # 张开端
    connected = False

    def command(self, value) -> None:
        """下发。value 长度是 dof,范围 [lo, hi]。"""

    def measured(self):
        """读实测。没有反馈就返回 None,上层会拿指令顶上并在数据里标出来。"""
        return None

    def close(self) -> None:
        pass

    # 数据里的列名,转 lerobot 时要用
    def names(self):
        return []


class NullEE(EndEffector):
    """什么都没有。state/action 就只有 7 个关节角。"""


class GripperEE(EndEffector):
    """两指夹爪,1 维,0=合拢 1=张开。

    **`port=None` 时不驱动任何硬件**,只把指令记下来。现在就是这个状态。
    """

    name = "gripper"
    dof = 1
    lo, hi = 0.0, 1.0

    def __init__(self, port=None, speed=1.0, force=1.0):
        self.port, self.speed, self.force = port, speed, force
        self._cmd = np.array([self.hi], dtype=float)     # 开机当作张开
        self._dev = None
        self.ticks = self.errors = 0
        self.last_error = ""
        if port:
            self._open()

    def _open(self):
        # 驱动还没写。接上线之后在这里开口子并把 connected 置 True。
        raise NotImplementedError(
            f"夹爪驱动还没写（port={self.port}）。先确认型号、通信方式、"
            f"有没有位置反馈，见 gripper.py 文件头。现在不给 --gripper-port "
            f"就是不驱动硬件、只记录指令的模式。")

    def command(self, value) -> None:
        v = np.clip(np.atleast_1d(np.asarray(value, dtype=float)), self.lo, self.hi)
        self._cmd = v[: self.dof]
        self.ticks += 1
        if self._dev is None:
            return                                        # 没接线：只记不发
        # 真驱动写在这里

    def measured(self):
        if self._dev is None:
            return None                                   # 没反馈，上层会标 ee_connected=False
        return None

    def cmd(self):
        return self._cmd.copy()

    def names(self):
        return ["gripper"]


class InspireHandEE(EndEffector):
    """因时 RH56,6 维,0~1000(0 合拢)。包一层,让它和夹爪长得一样。"""

    name = "hand"
    dof = 6
    lo, hi = 0.0, 1000.0

    FINGERS = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]

    def __init__(self, tracker):
        """tracker 是 avp_arm_teleop.HandTracker，手的指令由它自己的线程下发，
        这里只负责把它的快照翻译成统一接口。"""
        self.tracker = tracker
        self.connected = tracker is not None

    def command(self, value) -> None:
        pass                                              # 由 HandTracker 自己写

    def snapshot(self):
        """返回 (指令, 实测)。实测拿不到就 None。"""
        if self.tracker is None:
            return None, None
        cmd, meas, _ = self.tracker.snapshot_full()
        return cmd, meas

    def names(self):
        return list(self.FINGERS)


def pinch_to_grip(pinch_m, closed_m=0.015, open_m=0.075):
    """左手捏合距离(米) → 夹爪开度 0~1。

    端点给的是保守值：拇指食指指尖相触约 1.5cm（指尖有厚度，到不了 0），
    自然张开约 7.5cm。**现场要用 probe_axes.py 量一遍自己的手再填**，
    端点不对的话要么夹不拢，要么全程都在满开。
    """
    d = float(np.clip(pinch_m, closed_m, open_m))
    return (d - closed_m) / max(open_m - closed_m, 1e-6)


def make(kind, **kw) -> EndEffector:
    return {"none": NullEE, "gripper": GripperEE, "hand": InspireHandEE}[kind](**kw)
