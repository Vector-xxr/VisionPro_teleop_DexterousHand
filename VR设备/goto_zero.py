#!/usr/bin/env python3
"""把 AR5 送到**全部关节 0**（关节空间、排队式 MoveAbsJ）。

    python goto_zero.py                # 只报告：现在在哪、要走多少、法兰会经过哪里
    python goto_zero.py --go           # 真的动
    python goto_zero.py --go --payload 1.0    # 手装在臂上时才给负载

**全零不是收拢姿态，是整条臂竖直伸直。** 法兰跑到基座正上方 0.874 m，
比起始姿态高 624 mm。臂正上方有货架就会撞，而且**控制器层的碰撞检测是关掉的**
（config.py `collision_detection_enabled = False`），撞上去只剩 RSC 那道 17 Nm
关节力限制兜底。走之前自己看一眼头顶。

为什么不开 RT：和 `home_arm.py` 同理 —— 排队式 MoveAbsJ 由控制器自己规划，
比 RT 逐步伺服温和；关节空间不用逆解，从任何位形都出得来。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from ws_paths import HERE, SO101  # noqa: E402
sys.path.insert(0, str(SO101))   # atom_robot：AR5 驱动和 SO101 共用一份
sys.path.insert(0, str(HERE))

from atom_robot.ar5.ar5_env import AR5Env          # noqa: E402
from atom_robot.ar5.config import AR5Config        # noqa: E402


def report(q, cfg, tag):
    mg = cfg.joint_limit_margin
    lo = np.asarray(cfg.joint_min, float) + mg
    hi = np.asarray(cfg.joint_max, float) - mg
    m = np.minimum(q - lo, hi - q)
    k = int(np.argmin(m))
    print(f"  [{tag}] " + " ".join(f"j{i}:{np.rad2deg(v):+7.1f}°" for i, v in enumerate(q)))
    bad = [f"j{i}" for i in range(len(q)) if m[i] < 0]
    print(f"  [{tag}] 最紧 j{k} 余量 {np.rad2deg(m[k]):+.1f}°"
          + (f"   ⚠ **已在软限位外: {', '.join(bad)}**" if bad else ""))


def path_preview(q0, q1, n=11):
    """关节空间直线插值，看法兰会经过哪些高度。MoveAbsJ 走的就是关节空间。"""
    try:
        from ar5_poe_ik import AR5PoeIK
    except ImportError:
        print("  （没有本地运动学，跳过轨迹预览）")
        return None
    ik = AR5PoeIK()
    rows, zmax = [], -1e9
    for s in np.linspace(0.0, 1.0, n):
        T = ik.fk(q0 + (q1 - q0) * s)
        p = T[:3, 3]
        rows.append((s, p))
        zmax = max(zmax, float(p[2]))
    print(f"\n  法兰轨迹（关节空间直线插值，{n} 个采样点）:")
    for s, p in rows:
        bar = "█" * int(max(p[2], 0) * 30)
        print(f"    {s*100:3.0f}%  ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m  {bar}")
    return zmax


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="真的动。不加就只报告")
    ap.add_argument("--payload", type=float, default=0.0,
                    help="末端负载 kg。**手拆下来了就给 0**，给错了控制器会按不存在的"
                         "重量做重力补偿")
    ap.add_argument("--payload-com-z", type=float, default=0.08)
    ap.add_argument("--speed", type=float, default=0.0,
                    help="MoveAbsJ 速度，0=用 config 里的值。大行程建议给小一点")
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()

    c = AR5Config(mock=False, real_hand_in_mock=False)
    c.use_hand = False
    c.arm.use_realtime = False          # 不开 RT，走排队式
    c.cameras.sources = {}
    if a.speed > 0:
        c.arm.move_speed = a.speed

    target = np.zeros(c.arm.num_joints, dtype=float)
    lo = np.asarray(c.arm.joint_min, float) + c.arm.joint_limit_margin
    hi = np.asarray(c.arm.joint_max, float) - c.arm.joint_limit_margin
    if np.any(target < lo) or np.any(target > hi):
        print("  ✗ 全零在软限位外，这台臂到不了")
        return 2

    env = AR5Env(c)
    print("连接 AR5（只连，不开 RT）...", flush=True)
    env.connect()

    q0 = np.asarray(env.arm.get_joint_positions(), dtype=float)
    report(q0, c.arm, "当前")
    report(target, c.arm, "目标")
    d = np.rad2deg(target - q0)
    print(f"\n  每关节要走: " + " ".join(f"j{i}:{v:+7.1f}°" for i, v in enumerate(d)))
    print(f"  最大单关节行程 {np.max(np.abs(d)):.1f}° (j{int(np.argmax(np.abs(d)))})"
          f"   MoveAbsJ 速度 {c.arm.move_speed}")
    zmax = path_preview(q0, target)
    if zmax is not None:
        print(f"\n  法兰最高到 {zmax:.3f} m（基座系）。**臂正上方的净空自己确认**，"
              f"控制器层碰撞检测是关的。")
    print(f"  负载设定 {a.payload} kg" + ("（手拆下来了就该是 0）" if a.payload == 0 else ""))

    if not a.go:
        print("\n  这是预览，臂没动。确认头顶没东西之后加 --go")
        env.close()
        return 0

    r = env.arm._robot
    if hasattr(r, "setLoad"):
        try:
            ec = {}
            r.setLoad(env.arm.sdk.Load(a.payload, [0.0, 0.0, a.payload_com_z],
                                       [0.0, 0.0, 0.0]), ec)
            print(f"  [负载] {a.payload}kg 已设 (ec={ec.get('ec', 0)})")
        except Exception as e:                            # noqa: BLE001
            print(f"  [负载] 设不了({type(e).__name__})，沿用控制器里已存的设定")

    print("\n  ⚠ 臂即将移动到全零（排队式 MoveAbsJ）...", end=" ", flush=True)
    try:
        ok = env.goto_joints(target, blocking=True, timeout_s=a.timeout)
        print("完成" if ok is not False else "**未在超时内到位**")
    except Exception as e:                                # noqa: BLE001
        print(f"失败: {type(e).__name__}: {e}")
        env.close()
        return 1

    q1 = np.asarray(env.arm.get_joint_positions(), dtype=float)
    report(q1, c.arm, "到位后")
    err = np.rad2deg(np.abs(q1 - target))
    print(f"  与全零的偏差: 最大 {err.max():.2f}° (j{int(np.argmax(err))})")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
