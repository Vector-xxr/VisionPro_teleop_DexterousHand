#!/usr/bin/env python3
"""手动模式**点动**单个关节，把超出控制器软限位的关节挪回来。之后再跑 home_arm.py。

为什么不是拖动：drag_back.py 实测被控制器拒绝 —— ec=-10040「开始拖动失败,
正确设置负载并标定力矩传感器」（2026-10-09）。点动由控制器自己规划、只动一个关节，
是工业臂从超限位置出来的标准做法。

顺序照 SDK 官方 example/jog_example.py：
    NrtCommandMode → 手动模式 → 上电 → startJog(轴空间, 速率, 步长°, 关节号, 方向) → … → stop()
**点动开始后无论臂是否自己停下，都必须调 stop()**（官方 Note），否则一直处于 jog 状态。

    python jog_joint.py --dry-run          # 只报告：自动认出超限关节、方向、要转多少
    python jog_joint.py                    # 自动：超限关节往回转「还差的 + 3°」
    python jog_joint.py --joint 4 --deg 15 # 手动指定：j4 正向转 15°（负数 = 负向）
"""
from __future__ import annotations

import argparse
import select
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ws_paths import SO101  # noqa: E402
sys.path.insert(0, str(SO101))
from atom_robot.ar5.config import AR5Config  # noqa: E402
from drag_back import MARGIN_DEG, ec_msg, outside  # noqa: E402

EXTRA_DEG = 3.0


def main() -> int:
    ap = argparse.ArgumentParser(description="手动模式点动单关节")
    ap.add_argument("--joint", type=int, default=None, help="关节号 0~6；不给就自动认超限的那个")
    ap.add_argument("--deg", type=float, default=None, help="转多少度，正数=正向，负数=负向")
    ap.add_argument("--rate", type=float, default=0.05, help="速率 0.01~1（官方关节点动示例用 0.05）")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true", help="只报告，不改模式/电源，不动")
    args = ap.parse_args()
    if not 0.01 <= args.rate <= 0.2:
        raise SystemExit("--rate 限在 0.01~0.2（手动出限位不需要快）")

    cfg = AR5Config().arm
    sys.path.insert(0, cfg.sdk_path)
    sys.path.insert(0, str(Path(cfg.sdk_path) / "Release" / "linux"))
    import xCoreSDK_python as sdk

    print(f"连 {cfg.ip}（本地 {cfg.local_ip}）...")
    robot = getattr(sdk, cfg.robot_class)(cfg.ip, cfg.local_ip)

    def joints():
        return np.rad2deg(np.asarray(robot.jointPos({}), float))[:7]

    lim = sdk.PyTypeVectorArrayDouble2()
    robot.getSoftLimit(lim, {})
    lims = [tuple(np.rad2deg(v)) for v in lim.content()][:7]
    q = joints()
    print(f"  电源 {robot.powerState({})}   模式 {robot.operateMode({})}   运动状态 {robot.operationState({})}")
    print("  关节角 " + " ".join(f"j{i}:{v:+.1f}°" for i, v in enumerate(q)))

    if args.joint is None:
        rest = outside(q, lims, MARGIN_DEG)
        if not rest:
            print("  ✓ 没有关节超限或贴边，不用点动。直接 python home_arm.py")
            robot.disconnectFromRobot({})
            return 0
        if len(rest) > 1:
            print("  ⚠ 不止一个关节超限/贴边，一次只动一个，用 --joint/--deg 指定：")
            for i, cur, lo, hi, need, way in rest:
                print(f"     j{i} {cur:+.2f}° 限位[{lo:+.1f},{hi:+.1f}] 往{way}≥{need:.1f}°")
            robot.disconnectFromRobot({})
            return 1
        i, cur, lo, hi, need, way = rest[0]
        joint, deg = i, (need + EXTRA_DEG) * (1 if way.startswith("正") else -1)
    else:
        if args.deg is None:
            raise SystemExit("给了 --joint 就要给 --deg")
        joint, deg = args.joint, args.deg

    lo, hi = lims[joint]
    goal = q[joint] + deg
    print(f"\n  计划：j{joint} 从 {q[joint]:+.2f}° {'正' if deg > 0 else '负'}向转 {abs(deg):.1f}° "
          f"→ 约 {goal:+.2f}°   （限位 [{lo:+.1f}, {hi:+.1f}]，速率 {args.rate}）")
    if not (lo <= goal <= hi):
        print(f"  ✗ 转完仍在限位外，拒绝。")
        robot.disconnectFromRobot({})
        return 1
    if args.dry_run:
        print("\n--dry-run：只报告，没动任何东西。")
        robot.disconnectFromRobot({})
        return 0

    print("\n" + "=" * 64)
    print(f"  臂会动：只转 j{joint} 这一个关节，慢速。看着手上的网线和 24V 线。")
    print("  过程中按**回车**或 Ctrl+C 立即停。")
    print("=" * 64)
    if input("  确认后输入 yes 回车（其它输入取消）: ").strip().lower() != "yes":
        print("  取消，没改任何东西。")
        robot.disconnectFromRobot({})
        return 0

    jog_started = False
    try:
        for label, fn, *a in (("非实时模式", robot.setMotionControlMode, sdk.MotionControlMode.NrtCommandMode),
                              ("手动模式", robot.setOperateMode, sdk.OperateMode.manual),
                              ("上电", robot.setPowerState, True)):
            e = {}
            fn(*a, e)
            print(f"  {label} → {ec_msg(e)}")
            if e.get("ec", 0) != 0:
                print(f"  ✗ {label}失败，不点动。")
                return 1
        e = {}
        robot.startJog(sdk.JogOptSpace.jointSpace, args.rate, abs(deg), joint, deg > 0, e)
        print(f"  startJog(轴空间, {args.rate}, {abs(deg):.1f}°, j{joint}, {'正' if deg > 0 else '负'}向) → {ec_msg(e)}")
        if e.get("ec", 0) != 0:
            print("  ✗ 控制器拒绝点动，原话见上。")
            return 1
        jog_started = True

        t0 = time.time()
        still = 0
        prev = None
        while time.time() - t0 < args.timeout:
            v = joints()[joint]
            print(f"\r  j{joint} = {v:+8.2f}°   目标约 {goal:+.2f}°   {robot.operationState({})}".ljust(80),
                  end="", flush=True)
            # 判停：角度连续 1 秒（5 次）几乎不变，且已过最初 0.6 秒
            still = still + 1 if prev is not None and abs(v - prev) < 0.02 else 0
            prev = v
            if time.time() - t0 > 0.6 and still >= 5:
                break
            r, _, _ = select.select([sys.stdin], [], [], 0.2)
            if r:
                sys.stdin.readline()
                print("\n  手动停止")
                break
        print()
    except KeyboardInterrupt:
        print("\n  Ctrl+C")
    finally:
        if jog_started:
            e = {}
            robot.stop(e)           # 官方：点动后必须 stop()
            print(f"  stop() → {ec_msg(e)}")
        e = {}
        robot.setOperateMode(sdk.OperateMode.automatic, e)
        print(f"  回自动模式 → {ec_msg(e)}")
        q = joints()
        print("  关节角 " + " ".join(f"j{i}:{v:+.1f}°" for i, v in enumerate(q)))
        print(f"  电源 {robot.powerState({})}")
        left = outside(q, lims)
        robot.disconnectFromRobot({})
    if left:
        print("\n  ✗ 还有关节在限位外: " + ", ".join(f"j{b[0]}" for b in left))
        return 1
    print("\n  ✓ 都在限位内。下一步：python home_arm.py（看「回位后」那行关节角，别只看「完成」）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
