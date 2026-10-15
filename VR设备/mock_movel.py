#!/usr/bin/env python3
"""内存里的假 MoveL 控制器 —— 让 `--arm-backend movel` 能在 `--mock` 下跑完整条链路。

为什么要有这个：原来 `--mock` 碰到 `--arm-backend movel` 直接退回 rt 后端，
于是**主循环里 movel 那一整段分支从来没在离线环境里执行过**。和之前
「MockArm 没有 `_kinematics`，IK 注入那段代码从来没在 mock 里跑过」是同一类
盲区 —— 那次的代价是一个 `else` 分支把求解器覆盖掉、横幅照常打印，没人发现。

所以这个假控制器不是"桩"，它**真的走队列**：
  · `moveAppend` 把路点排进队列
  · `moveStart` 开始消费
  · 队列按 `speed` 沿折线推进当前位置，走完自动转 idle
  · `operationState` 据此回 moving / idle

这样前导闸门、抽稀、队列跑空补 moveStart、`_target_pose` 同步这些**时序相关**
的逻辑在离线就能被真正触发，而不是等上真机才第一次执行。

⚠ 它不模拟：逆解失败、限位、奇异、加速度受限（真机上短段跑不到指令速度，
实测只有 25~33mm/s，这里是理想匀速）。所以它验的是**控制流**，不是运动学。
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np


class MockMoveLSdk:
    """只提供 MoveLStreamer 用到的那几个符号。"""

    class CoordinateType:
        flangeInBase = "flangeInBase"

    class OperationState:
        moving = "moving"
        idle = "idle"
        unknown = "unknown"

    class MotionControlMode:
        NrtCommandMode = "nrt"
        RtCommandMode = "rt"

    class OperateMode:
        automatic = "automatic"

    class PowerState:
        on = "PowerState.on"
        off = "PowerState.off"

    class CartesianPosition:
        def __init__(self, *_a):
            self.trans = [0.0, 0.0, 0.0]
            self.rpy = [0.0, 0.0, 0.0]
            self.elbow = 0.0
            self.hasElbow = False
            self.confData = [0] * 8

    class Frame:
        def __init__(self):
            self.trans, self.rpy = [0.] * 3, [0.] * 3

    class Load:
        def __init__(self, mass, cog, inertia):
            self.mass, self.cog, self.inertia = mass, cog, inertia

    class MoveLCommand:
        def __init__(self, target):
            self.target = target
            self.speed = 0.0
            self.zone = 0.0

    @staticmethod
    def PyString(s):
        return s


class MockMoveLRobot:
    """假控制器：维护一条路点队列，按 speed 沿折线推进。"""

    def __init__(self, speed_mm_s: float = 250.0, start=None, rpy=None):
        self.speed = float(speed_mm_s) / 1000.0          # m/s
        self.pos = np.zeros(3) if start is None else np.asarray(start, dtype=float).copy()
        self.rpy = [0.0, 0.0, 0.0] if rpy is None else list(rpy)
        self.queue: deque = deque()
        self.running = False
        self._t = time.perf_counter()
        # 给调用方看的计数，mock 跑完可以核对
        self.n_append = 0          # 收到的**点**数
        self.n_append_calls = 0    # moveAppend 的**调用**次数（批量时 < n_append）
        self.n_start = 0
        self.n_reset = 0
        self.max_queue = 0
        self.powered = True        # 真控制器可能是关的；ensure_power 关了会去上电
        from types import SimpleNamespace
        self._toolset = SimpleNamespace(end=MockMoveLSdk.Frame(), ref=MockMoveLSdk.Frame(), load=None)

    # ── 内部：把时间推进到现在 ──────────────────────────────────────────
    def _advance(self):
        now = time.perf_counter()
        dt, self._t = now - self._t, now
        if not self.running or dt <= 0:
            return
        # ⚠ 速度要用**这一段自己的** speed，不是机器人的默认速度。
        # 第一版在这里用了 self.speed，结果 --movel-adaptive-speed 在 mock 里
        # 完全没有效果（段速从 250 降到 58，队列跑空次数一次都没少）——
        # 假控制器把要测的那个变量给忽略了，等于测了个寂寞。
        remaining = dt
        while remaining > 0 and self.queue:
            tgt, seg_v = self.queue[0]
            v = max(float(seg_v), 1e-6) / 1000.0        # mm/s → m/s
            d = float(np.linalg.norm(tgt - self.pos))
            need = d / v
            if need <= remaining:
                self.pos = tgt.copy()
                self.queue.popleft()
                remaining -= need
            else:
                self.pos = self.pos + (tgt - self.pos) / d * (v * remaining)
                remaining = 0.0
        if not self.queue:
            # 队列走光 → 臂停 → 状态退 moving。真机上状态比位置晚 87ms，
            # 这里不模拟那个滞后（方向是保守的：真机更不容易误判 idle）。
            self.running = False

    # ── MoveLStreamer 用到的接口 ────────────────────────────────────────
    def cartPosture(self, _coord, ec):
        self._advance()
        ec["ec"] = 0
        p = MockMoveLSdk.CartesianPosition()
        p.trans = list(self.pos)
        p.rpy = list(self.rpy)
        p.confData = [0] * 8
        return p

    def operationState(self, ec):
        self._advance()
        ec["ec"] = 0
        return (MockMoveLSdk.OperationState.moving if self.running
                else MockMoveLSdk.OperationState.idle)

    def moveAppend(self, cmd, _name, ec):
        """两种形态都收，和真 SDK 的重载对齐：

            #2  moveAppend(cmds: list[MoveLCommand], cmdID, ec)
            #8  moveAppend(cmd:  MoveLCommand,       cmdID, ec)

        假控制器必须两种都支持，否则「改走批量」这个改动在离线根本测不了 ——
        上一次就是因为假控制器忽略了 cmd.speed，自适应段速在 mock 里假装没效果。
        """
        self._advance()
        cmds = cmd if isinstance(cmd, (list, tuple)) else [cmd]
        for c in cmds:
            self.queue.append((np.asarray(c.target.trans, dtype=float),
                               float(getattr(c, "speed", 0.0) or self.speed * 1000.0)))
            self.rpy = list(c.target.rpy)
            self.n_append += 1
        self.n_append_calls += 1
        self.max_queue = max(self.max_queue, len(self.queue))
        ec["ec"] = 0

    def moveStart(self, ec):
        self._advance()
        self.n_start += 1
        if self.running:
            ec["ec"] = -20                      # 良性：机器人运动中
            ec["message"] = "机器人运动中"
            return
        if not self.queue:
            ec["ec"] = 768                      # 良性：队列已空
            ec["message"] = "没有可执行的运动指令"
            return
        self.running = True
        ec["ec"] = 0

    def moveReset(self, ec):
        self.n_reset += 1
        self.queue.clear()
        self.running = False
        ec["ec"] = 0

    def stop(self, ec):
        self._advance()
        self.running = False
        ec['ec'] = 0

    def toolset(self, ec):
        import copy
        ec['ec'] = 0
        return copy.deepcopy(self._toolset)

    def setToolset(self, toolset, ec):
        import copy
        self._toolset = copy.deepcopy(toolset)
        ec['ec'] = 0

    # 配置步骤，全部 no-op 成功
    def setMotionControlMode(self, _m, ec):
        ec["ec"] = 0

    def setOperateMode(self, _m, ec):
        ec["ec"] = 0

    def setPowerState(self, on, ec):
        self.powered = bool(on)
        ec["ec"] = 0

    def powerState(self, ec):
        ec["ec"] = 0
        return MockMoveLSdk.PowerState.on if self.powered else MockMoveLSdk.PowerState.off

    def setDefaultConfOpt(self, _v, ec):
        ec["ec"] = 0

    def setMaxCacheSize(self, _n, ec):
        ec["ec"] = 0

    def setDefaultSpeed(self, _v, ec):
        ec["ec"] = 0

    def setDefaultZone(self, _v, ec):
        ec["ec"] = 0

    def report(self) -> str:
        return (f"假控制器: 收到 {self.n_append} 点 / {self.n_append_calls} 次 moveAppend  moveStart {self.n_start} 次  "
                f"队列峰值 {self.max_queue}  剩余 {len(self.queue)}  "
                f"位置 {np.round(self.pos * 1000, 1).tolist()} mm")
