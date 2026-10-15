#!/usr/bin/env python3
"""采数据:把一条 episode 攒在内存里,结束时后台写盘。

每条 episode 存成一个 `.npz` + 每路相机一个 `.mp4`。

**写盘必须在后台线程上。** servoJ 要求每 8ms 收到一个关节角,主循环停 2 秒就会被
控制器判成通信丢包。实测 600 帧两路相机写一次要 2 秒出头。

字段。维度 = 7 个关节角 + 末端执行器的维数,末端是什么由 `ee_dof` 决定:

    末端      ee_dof   state/action 维数   范围
    夹爪        1            8            0~1     (0 合拢)
    因时手      6           13            0~1000  (0 合拢)
    无          0            7            —

    state      实测:7 个关节角(rad) + 末端实测
    action     指令:同上。servoJ 下发的就是绝对关节角,末端也是绝对指令,
               所以动作空间用绝对值,不用增量
    ee_pose     6   实测末端绝对位姿:位置(m) + 轴角(rad)
    ee_target   6   指令末端位姿。留着以后想换动作空间时重算
    timestamp   n   每帧相对第一帧的秒数,实测。**转数据集用它,不要用标称 fps**
    ee_connected    末端硬件在不在。**False 时 state 的末端维是拿指令顶上的**,
                    不是真实测值,训练时要知道这一点
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

ARM_DOF = 7


class EpisodeRecorder:
    def __init__(self, out_dir, camera_keys, fps=30.0, task="", codec="mp4v",
                 min_frames=60, min_ee_path_mm=50.0,
                 ee_dof=1, ee_names=None, ee_range=(0.0, 1.0), ee_connected=False):
        self.out_dir = Path(out_dir).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.camera_keys = list(camera_keys)
        self.fps, self.task, self.codec = float(fps), task, codec
        self.min_frames, self.min_ee_path_mm = int(min_frames), float(min_ee_path_mm)
        self.ee_dof = int(ee_dof)
        self.ee_names = list(ee_names) if ee_names else [f"ee{i}" for i in range(self.ee_dof)]
        self.ee_lo, self.ee_hi = float(ee_range[0]), float(ee_range[1])
        self.ee_connected = bool(ee_connected)
        self.state_dim = ARM_DOF + self.ee_dof
        self._writer = None
        self._write_error = None
        self.reset()

    def reset(self):
        self.state, self.action = [], []
        self.ee_pose, self.ee_target = [], []
        self.stamps = []
        self.frames = {k: [] for k in self.camera_keys}
        self.dropped_frames = 0          # 相机没给新图的次数
        self.faked_meas = 0              # 末端没反馈、拿指令顶上的帧数

    def __len__(self):
        return len(self.state)

    # ── 录一帧 ────────────────────────────────────────────────────────────
    def add(self, q_meas, q_cmd, ee_meas, ee_cmd, ee_pose, ee_target, images):
        """images: {相机名: RGB uint8 图}。缺某一路就跳过这一帧,不留空洞。

        `ee_meas` 是 None 表示末端没有位置反馈(或者根本没接)。这时拿指令顶上,
        并计数 —— 存盘时 `ee_connected` 会如实标出来。
        """
        for k in self.camera_keys:
            if images is None or images.get(k) is None:
                self.dropped_frames += 1
                return False
        self.stamps.append(time.perf_counter())

        def _ee(v):
            if self.ee_dof == 0:
                return np.zeros(0)
            return np.asarray(v, dtype=float).reshape(-1)[: self.ee_dof]

        ec = _ee(np.zeros(self.ee_dof) if ee_cmd is None else ee_cmd)
        if ee_meas is None:
            em = ec.copy()                       # 没反馈：指令顶上
            if self.ee_dof:
                self.faked_meas += 1
        else:
            em = _ee(ee_meas)

        self.state.append(np.concatenate([np.asarray(q_meas, float)[:ARM_DOF], em]).astype(np.float32))
        self.action.append(np.concatenate([np.asarray(q_cmd, float)[:ARM_DOF], ec]).astype(np.float32))
        self.ee_pose.append(np.asarray(ee_pose, dtype=np.float32))
        self.ee_target.append(np.asarray(ee_target, dtype=np.float32))
        for k in self.camera_keys:
            self.frames[k].append(images[k])
        return True

    def measured_fps(self):
        if len(self.stamps) < 2:
            return None
        span = self.stamps[-1] - self.stamps[0]
        return (len(self.stamps) - 1) / span if span > 1e-6 else None

    # ── 存之前看一眼这条能不能用 ──────────────────────────────────────────
    def warnings(self):
        n = []
        if len(self) < self.min_frames:
            n.append(f"只有 {len(self)} 帧,短于 {self.min_frames}")
        if len(self) < 2:
            return n
        a = np.stack(self.action)
        if self.ee_dof:
            ee = a[:, ARM_DOF:]
            span = float(np.max(ee.max(0) - ee.min(0)))
            # 阈值按量程的 2% 定，夹爪(0~1)和手(0~1000)用同一条规则
            if span < (self.ee_hi - self.ee_lo) * 0.02:
                n.append("末端全程几乎没动,那几维学不到东西")
        p = np.stack(self.ee_pose)[:, :3]
        path = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) * 1000
        if path < self.min_ee_path_mm:
            n.append(f"末端一共只走了 {path:.0f} mm(阈值 {self.min_ee_path_mm:.0f})")
        m = self.measured_fps()
        if m and abs(m - self.fps) > self.fps * 0.15:
            n.append(f"实测帧率 {m:.1f} Hz,标称 {self.fps:.0f} Hz,差 {abs(m-self.fps)/self.fps*100:.0f}%")
        if self.dropped_frames:
            n.append(f"相机少给了 {self.dropped_frames} 帧(那些时刻整帧丢弃,不影响已存的)")
        if self.faked_meas:
            n.append(f"末端没有位置反馈,{self.faked_meas} 帧的实测值是拿指令顶的")
        return n

    # ── 落盘 ──────────────────────────────────────────────────────────────
    def save(self, index, task=None, success=True):
        """把缓冲交给后台线程,立刻返回。缓冲当场换新的,可以接着录。"""
        self.flush()
        st = np.asarray(self.stamps, dtype=np.float64)
        payload = dict(
            state=np.stack(self.state),
            action=np.stack(self.action),
            ee_pose=np.stack(self.ee_pose),
            ee_target=np.stack(self.ee_target),
            task=task if task is not None else self.task,
            fps=self.fps,
            timestamp=st - st[0] if len(st) else st,
            fps_measured=float(self.measured_fps() or self.fps),
            camera_keys=np.array(self.camera_keys),
            arm_dof=ARM_DOF,
            ee_dof=self.ee_dof,
            ee_names=np.array(self.ee_names),
            ee_range=np.array([self.ee_lo, self.ee_hi]),
            # 末端硬件在不在。False 时 state 的末端维是拿指令顶的,不是真实测。
            ee_connected=bool(self.ee_connected and not self.faked_meas),
            # 这条算不算做成了。训练时按它筛,不用靠人记。
            success=bool(success),
        )
        frames, self.frames = self.frames, {k: [] for k in self.camera_keys}
        path = self.out_dir / f"episode_{index:04d}.npz"
        self.reset()
        self._writer = threading.Thread(target=self._write, args=(path, payload, frames),
                                        name=f"rec-{index:04d}", daemon=False)
        self._writer.start()
        return path

    def _write(self, path, payload, frames):
        try:
            import cv2
            # 写视频的帧率要**钳一下**。原来直接用实测值，回路跑得飞快时（比如离线
            # 造数据、或者相机没接只跑逻辑）实测能到几千 Hz，MPEG-4 的时基放不下，
            # VideoWriter 打不开 —— 而它**不抛异常**，于是 mp4 是空的、npz 却好好的，
            # 要到转数据集时才发现「有 npz 没视频」。钳完再对不上就退回标称值。
            fps = float(payload["fps_measured"])
            if not (1.0 <= fps <= 240.0):
                self._write_error = (f"{path.name}: 实测帧率 {fps:.0f}Hz 不合理，"
                                     f"视频按标称 {self.fps:.0f}Hz 写")
                fps = float(min(max(self.fps, 1.0), 240.0))
            for key, imgs in frames.items():
                if not imgs:
                    continue
                h, w = imgs[0].shape[:2]
                out = path.with_name(f"{path.stem}_{key}.mp4")
                vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*self.codec),
                                     fps, (w, h))
                if not vw.isOpened():        # 打不开就说出来，别留个空文件
                    raise RuntimeError(f"VideoWriter 打不开 {out.name}"
                                       f"（{w}x{h} @{fps:.1f}Hz codec={self.codec}）")
                for f in imgs:
                    vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))   # 驱动给 RGB,写视频要 BGR
                vw.release()
            np.savez_compressed(path, **payload)
        except Exception as exc:                # noqa: BLE001
            self._write_error = f"{path.name} 写盘失败:{exc}"

    def flush(self, timeout=90.0):
        if self._writer is not None:
            self._writer.join(timeout)
            if self._writer.is_alive():
                print(f"  ⚠ 上一条还在写盘(超过 {timeout:.0f}s)")
            self._writer = None
        if self._write_error:
            print(f"  ✗ {self._write_error}")
            self._write_error = None


class LayoutError(ValueError):
    """这个文件不是本流水线的格式,读不了。"""


def read_layout(npz):
    """从一条数据里读出它的维度约定。**老数据没有 ee_dof,按 13 维手来认。**

    认不出来就抛 `LayoutError`,上层跳过这个文件并说明白,不要崩在某个
    zero-size 的 reduce 上 —— 那种报错完全看不出是格式不对。
    """
    arm = int(npz["arm_dof"]) if "arm_dof" in npz else ARM_DOF
    if "ee_dof" in npz:
        dof = int(npz["ee_dof"])
        names = [str(x) for x in npz["ee_names"]] if "ee_names" in npz else []
        lo, hi = (float(npz["ee_range"][0]), float(npz["ee_range"][1])) \
            if "ee_range" in npz else (0.0, 1.0)
    elif "hand_dof" in npz:                       # 换夹爪之前的数据
        dof, lo, hi = int(npz["hand_dof"]), 0.0, 1000.0
        names = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"][:dof]
    else:
        dof, lo, hi, names = int(len(npz["state"][0])) - arm, 0.0, 1000.0, []
    if not names:
        names = ["gripper"] if dof == 1 else [f"ee{i}" for i in range(dof)]
    connected = bool(npz["ee_connected"]) if "ee_connected" in npz else True

    # state 和 action 都必须是 arm + dof 宽。对不上就不是这套格式 ——
    # 比如 VEL 那批老数据是**增量**动作空间(6 个位姿增量 + 1 个夹爪),
    # action 只有 7 列而 state 有 8 列,和这里的绝对关节角完全不是一回事。
    ns, na = int(np.shape(npz["state"])[1]), int(np.shape(npz["action"])[1])
    if ns != arm + dof or na != arm + dof:
        raise LayoutError(
            f"state {ns} 列 / action {na} 列，按 {arm}+{dof} 读不出来。"
            f"这多半是别的流水线的数据（比如增量动作空间的老数据），"
            f"不要和这批混在一个目录里")
    return arm, dof, names, (lo, hi), connected


def next_index(out_dir):
    """已有几条就从几号接着编,不覆盖。"""
    out_dir = Path(out_dir).expanduser()
    if not out_dir.is_dir():
        return 0
    used = [int(p.stem.split("_")[1]) for p in out_dir.glob("episode_*.npz")
            if p.stem.split("_")[1].isdigit()]
    return max(used) + 1 if used else 0
