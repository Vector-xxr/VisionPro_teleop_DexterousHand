#!/usr/bin/env python3
"""关节超出控制器软限位后，用**拖动**把它挪回来。之后再跑 home_arm.py 回位。

为什么需要：关节一旦在控制器软限位外（2026-10-09 servoJ 遥操把 j4 甩到 -184.95°，
限位 ±178°），自动模式下**任何运动都会被静默拒绝** —— moveExecution 报
reachTarget=False、ec=0，home_arm.py 还会误报「完成」。只有手动模式拖动/点动能出来。

⚠ 2026-10-09 实测**这台臂现在拖不了**：enableDrag 返回 ec=-10040「开始拖动失败,
正确设置负载并标定力矩传感器」。要先设负载 + 标定力矩传感器。出限位改用 jog_joint.py（点动）。

顺序照 SDK 官方 example/drag_example.py：
    下电 → 手动模式 → moveReset → enableDrag(轴空间, 自由拖拽)
「打开拖动之前，需要机械臂处于手动模式下电状态」是官方原注释。
结束：disableDrag → 回自动模式 → 保持下电 → 断开。上电交给 home_arm.py。

    python drag_back.py --dry-run   # 只报告：哪个关节超限、往哪边转、差多少。不改任何东西
    python drag_back.py             # 真的打开拖动（要输入 yes）
"""
from __future__ import annotations

import argparse
import select
import sys
import time
from pathlib import Path

import numpy as np

from ws_paths import SO101  # noqa: E402
sys.path.insert(0, str(SO101))
from atom_robot.ar5.config import AR5Config  # noqa: E402

MARGIN_DEG = 5.0          # 回到限位内多少度才算安全（home_arm 走的是带 2.9° 余量的软带）


def ec_msg(ec: dict) -> str:
    return f"ec={ec.get('ec', 0)} {ec.get('message', '')}"


def outside(q_deg, lims_deg, margin=0.0):
    """返回 [(关节号, 当前, 下限, 上限, 还差多少度, 往哪边转)]，margin 是要求的内缩量。"""
    out = []
    for i, (lo, hi) in enumerate(lims_deg):
        if q_deg[i] < lo + margin:
            out.append((i, q_deg[i], lo, hi, lo + margin - q_deg[i], "正方向（往大转）"))
        elif q_deg[i] > hi - margin:
            out.append((i, q_deg[i], lo, hi, q_deg[i] - (hi - margin), "负方向（往小转）"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="拖动把超限关节挪回软限位内")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不改模式/电源，不打开拖动")
    args = ap.parse_args()

    cfg = AR5Config().arm
    sys.path.insert(0, cfg.sdk_path)
    sys.path.insert(0, str(Path(cfg.sdk_path) / "Release" / "linux"))
    import xCoreSDK_python as sdk

    print(f"连 {cfg.ip}（本地 {cfg.local_ip}）...")
    robot = getattr(sdk, cfg.robot_class)(cfg.ip, cfg.local_ip)
    ec = {}
    n = 7

    def joints():
        return np.rad2deg(np.asarray(robot.jointPos({}), float))[:n]

    lim = sdk.PyTypeVectorArrayDouble2()
    enabled = robot.getSoftLimit(lim, {})
    lims = [tuple(np.rad2deg(v)) for v in lim.content()][:n]
    print(f"  电源 {robot.powerState(ec)}   模式 {robot.operateMode(ec)}   运动状态 {robot.operationState(ec)}")
    print(f"  控制器软限位 开启={enabled}")
    q = joints()
    bad = outside(q, lims)
    for i, cur, lo, hi, need, way in outside(q, lims, MARGIN_DEG):
        tag = "超限" if any(b[0] == i for b in bad) else "贴边"
        print(f"  ⚠ j{i} {tag}: 当前 {cur:+.2f}°  限位 [{lo:+.1f}, {hi:+.1f}]  "
              f"→ 往{way}再转 ≥{need:.1f}°")
    if not outside(q, lims, MARGIN_DEG):
        print("  ✓ 所有关节都在限位内且留够余量，不用拖。直接 python home_arm.py")
        robot.disconnectFromRobot({})
        return 0

    if args.dry_run:
        print("\n--dry-run：只报告，没动任何东西。")
        robot.disconnectFromRobot({})
        return 0

    print("\n" + "=" * 64)
    print("  就要打开**关节空间自由拖动**。拖动时臂没有自己的位置保持：")
    print("   · 手（约 1 kg）的重量控制器可能没算进去，松手时手腕/前臂会往下坠 ——")
    print("     **全程扶住手腕**，结束拖动（回车）之前别松手")
    print("   · 只拧超限的那个关节，按上面提示的方向；转过头会进另一侧的限位")
    print("   · 看着手上的网线和 24V 线，别拧缠")
    print("=" * 64)
    if input("  确认后输入 yes 回车（其它输入取消）: ").strip().lower() != "yes":
        print("  取消，没改任何东西。")
        robot.disconnectFromRobot({})
        return 0

    drag_on = False
    try:
        if str(robot.operationState({})).endswith("rtControlling"):
            e = {}
            robot.setMotionControlMode(sdk.MotionControlMode.NrtCommandMode, e)
            print(f"  清掉残留 rtControlling → {ec_msg(e)}")
        for label, fn, *a in (("下电", robot.setPowerState, False),
                              ("手动模式", robot.setOperateMode, sdk.OperateMode.manual),
                              ("moveReset", robot.moveReset)):
            e = {}
            fn(*a, e)
            print(f"  {label} → {ec_msg(e)}")
        e = {}
        robot.enableDrag(sdk.DragParameterSpace.jointSpace, sdk.DragParameterType.freely, e, True)
        print(f"  打开拖动（轴空间、自由拖拽、无需按末端键）→ {ec_msg(e)}")
        if e.get("ec", 0) != 0:
            print("  ✗ 控制器拒绝打开拖动，原话见上。没有进入拖动状态。")
            return 1
        drag_on = True

        print("\n  ▶ 现在扶住手腕，把超限关节往提示的方向拧。回到限位内会提示。"
              "\n    拧好后按**回车**结束拖动（Ctrl+C 也会安全关闭）。\n")
        done_shown = False
        while True:
            q = joints()
            rest = outside(q, lims, MARGIN_DEG)
            line = "  ".join(f"j{i} {cur:+7.2f}° 还差 {need:4.1f}°" for i, cur, _, _, need, _ in rest)
            if rest:
                msg = f"\r  {line}   "
                done_shown = False
            else:
                msg = "\r  ✓ 已全部回到限位内（留了 ≥5° 余量）—— 扶住，按回车结束拖动"
                if not done_shown:
                    print()
                    done_shown = True
            print(msg.ljust(100), end="", flush=True)
            r, _, _ = select.select([sys.stdin], [], [], 0.2)
            if r:
                sys.stdin.readline()
                break
        print()
    except KeyboardInterrupt:
        print("\n  Ctrl+C")
    finally:
        if drag_on:
            e = {}
            robot.disableDrag(e)
            print(f"  关闭拖动 → {ec_msg(e)}")
        e = {}
        robot.setOperateMode(sdk.OperateMode.automatic, e)
        print(f"  回自动模式 → {ec_msg(e)}")
        q = joints()
        left = outside(q, lims)
        print("  关节角 " + " ".join(f"j{i}:{v:+.1f}°" for i, v in enumerate(q)))
        # 实测：先 setPowerState(False) 成功、拖动被拒、回自动模式后这里读到的是 on，
        # 所以不声称「已下电」，如实打印
        print(f"  电源 {robot.powerState({})}")
        robot.disconnectFromRobot({})
    if left:
        print("\n  ✗ 还有关节在限位外: " + ", ".join(f"j{b[0]}" for b in left) + "。再跑一次。")
        return 1
    print("\n  下一步：python home_arm.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
