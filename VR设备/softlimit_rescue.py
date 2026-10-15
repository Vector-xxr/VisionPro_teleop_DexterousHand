#!/usr/bin/env python3
"""**临时关软限位**，只把超限关节转回限位内，然后**无论成败都恢复软限位**。之后跑 home_arm.py。

为什么要这一招（2026-10-09）：servoJ 遥操把 j4 甩到 -184.95°（控制器软限位 ±178°），
之后自动模式 MoveAbsJ 被静默拒绝（reachTarget=False、ec=0），拖动被拒（ec=-10040
要先设负载+标定力矩传感器），手动点动 startJog 返回 ec=0、状态变 jogging/moving 但关节一动不动。
用户 2026-10-09 明确同意走这条：临时关掉软限位这道保护。

把「关着保护」的窗口压到最小：
  · 只动超限的关节，其余关节目标 = 当前值（原地）
  · 只转回限位内 --margin 度（默认 13° → j4 到 -165°），不直接回 home
  · 默认**全程手动模式点动**（--rate 0.05）。自动模式 MoveAbsJ（--via movej）实测在软限位关着时
    moveStart 一直 -17「该操作不允许在当前上下电状态下执行」
  · 收尾固定顺序：stop() → 手动 → 下电 → setSoftLimit(True) → 自动（改软限位只许在手动模式，自动模式 -16）
  · finally 里 setSoftLimit(True) **不传数值** —— SDK 文档：「limits 为默认值时仅打开软限位不修改数值」、
    「关闭软限位时不会修改限位数值」—— 然后读回来逐轴比对关之前存下的值

    python softlimit_rescue.py --dry-run   # 只报告计划，不改任何东西
    python softlimit_rescue.py             # 真做（要输入 yes）
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
from drag_back import ec_msg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="临时关软限位把超限关节转回来，结束必恢复")
    ap.add_argument("--margin", type=float, default=13.0, help="转回限位内多少度")
    ap.add_argument("--via", choices=("jog", "movej"), default="jog",
                    help="jog=手动模式点动（默认）；movej=自动模式 MoveAbsJ（软限位关着时实测 -17）")
    ap.add_argument("--rate", type=float, default=0.05, help="点动速率 0.01~1（官方关节点动示例 0.05）")
    ap.add_argument("--speed", type=float, default=100.0, help="MoveAbsJ 速度参数（驱动平时 200）")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not 0.01 <= args.rate <= 0.2:
        raise SystemExit("--rate 限在 0.01~0.2（软限位关着，只许慢）")

    cfg = AR5Config().arm
    sys.path.insert(0, cfg.sdk_path)
    sys.path.insert(0, str(Path(cfg.sdk_path) / "Release" / "linux"))
    import xCoreSDK_python as sdk

    print(f"连 {cfg.ip}（本地 {cfg.local_ip}）...")
    robot = getattr(sdk, cfg.robot_class)(cfg.ip, cfg.local_ip)

    def joints_rad():
        return np.asarray(robot.jointPos({}), float)[:7]

    def soft_limits():
        lim = sdk.PyTypeVectorArrayDouble2()
        en = robot.getSoftLimit(lim, {})
        return bool(en), np.asarray([list(v) for v in lim.content()][:7], float)

    enabled0, lims0 = soft_limits()                 # 关之前存下，最后逐轴比对
    q = joints_rad()
    print(f"  电源 {robot.powerState({})}   模式 {robot.operateMode({})}   运动状态 {robot.operationState({})}")
    print(f"  软限位 开启={enabled0}")
    goal = q.copy()
    bad = []
    m = np.deg2rad(args.margin)
    for i in range(7):
        lo, hi = lims0[i]
        if q[i] < lo:
            goal[i] = lo + m
            bad.append(i)
        elif q[i] > hi:
            goal[i] = hi - m
            bad.append(i)
    if not bad:
        print("  ✓ 没有关节在软限位外，不需要关保护。直接 python home_arm.py")
        robot.disconnectFromRobot({})
        return 0
    if not enabled0:
        print("  ⚠ 软限位本来就是关的 —— 这个脚本结束时会把它**打开**。")
    for i in bad:
        print(f"  计划：j{i} {np.rad2deg(q[i]):+.2f}° → {np.rad2deg(goal[i]):+.2f}°   "
              f"（限位 [{np.rad2deg(lims0[i][0]):+.1f}, {np.rad2deg(lims0[i][1]):+.1f}]，其余关节原地）")
    if args.dry_run:
        print("\n--dry-run：只报告，没动任何东西。")
        robot.disconnectFromRobot({})
        return 0

    print("\n" + "=" * 66)
    print("  ⚠ 会**临时关掉控制器软限位**，然后臂会动（只转上面那个关节，慢速）。")
    print("    结束时（含出错/Ctrl+C）一定重新打开并核对。过程中按回车或 Ctrl+C 停。")
    print("    看着手上的网线和 24V 线。")
    print("=" * 66)
    if input("  确认后输入 yes 回车（其它输入取消）: ").strip().lower() != "yes":
        print("  取消，没改任何东西。")
        robot.disconnectFromRobot({})
        return 0

    def step(label, fn, *a):
        e = {}
        fn(*a, e)
        print(f"  {label} → {ec_msg(e)}")
        return e.get("ec", 0)

    limits_off = False
    moved = False
    try:
        st = str(robot.operationState({}))
        if st.endswith(("jogging", "moving")):
            step("先停掉残留运动 stop()", robot.stop)
        if step("关软限位", robot.setSoftLimit, False) != 0:
            # 有的控制器要求手动模式+下电才能改软限位，按报错再试一次
            step("手动模式", robot.setOperateMode, sdk.OperateMode.manual)
            step("下电", robot.setPowerState, False)
            if step("关软限位（重试）", robot.setSoftLimit, False) != 0:
                print("  ✗ 控制器不让关软限位，原话见上。没有动臂。")
                return 1
        limits_off = True
        en, _ = soft_limits()
        print(f"  读回 软限位开启={en}")

        # 模式：jog（默认）= 全程手动模式点动；movej = 自动模式 MoveAbsJ。
        # 2026-10-09 实测：软限位关着时，自动模式 moveStart 连续 -17「该操作不允许在当前上下电状态下执行」
        # （上电已生效、等了 1.5 s 照样），所以默认改成手动点动 —— 控制器提示原话就是
        # 「轴角度超出运动范围, 尝试取消软限位后恢复各轴到允许的范围内」，改软限位本身也只许在手动模式。
        mode = sdk.OperateMode.manual if args.via == "jog" else sdk.OperateMode.automatic
        for label, fn, *a in (("非实时模式", robot.setMotionControlMode, sdk.MotionControlMode.NrtCommandMode),
                              (f"{'手动' if args.via == 'jog' else '自动'}模式", robot.setOperateMode, mode),
                              ("上电", robot.setPowerState, True)):
            if step(label, fn, *a) != 0:
                print(f"  ✗ {label}失败，不发运动。")
                return 1
        # 上电是异步的：setPowerState 返回 0 只代表指令收到了。等 powerState 真变 on，再给伺服 1.5 s。
        t_pw = time.time()
        while time.time() - t_pw < 10.0 and not str(robot.powerState({})).endswith(".on"):
            time.sleep(0.1)
        print(f"  等上电生效 {time.time() - t_pw:.1f}s → {robot.powerState({})}，再稳 1.5s")
        time.sleep(1.5)
        if step("moveReset", robot.moveReset) != 0:
            print("  ✗ moveReset 失败，不发运动。")
            return 1

        if args.via == "jog":
            if len(bad) != 1:
                print(f"  ✗ 点动一次只动一个关节，现在超限的有 {bad}。用 --via movej，或先处理一个。")
                return 1
            j = bad[0]
            deg = float(np.rad2deg(goal[j] - q[j]))
            code = step(f"startJog(轴空间, {args.rate}, {abs(deg):.1f}°, j{j}, {'正' if deg > 0 else '负'}向)",
                        robot.startJog, sdk.JogOptSpace.jointSpace, args.rate, abs(deg), j, deg > 0)
        else:
            cmd = sdk.MoveAbsJCommand(sdk.JointPosition(goal.tolist()), args.speed, 0.0)
            if step("moveAppend MoveAbsJ", robot.moveAppend, [cmd], sdk.PyString("softlimit_rescue")) != 0:
                return 1
            for attempt in range(3):
                code = step(f"moveStart（第 {attempt + 1} 次）", robot.moveStart)
                if code != -17:
                    break
                time.sleep(0.5)
        if code != 0:
            return 1
        moved = True

        t0 = time.time()
        while time.time() - t0 < args.timeout:
            qd = np.rad2deg(joints_rad())
            err = np.max(np.abs(qd[bad] - np.rad2deg(goal[bad])))
            st = robot.operationState({})
            print(f"\r  " + "  ".join(f"j{i} {qd[i]:+8.2f}°" for i in bad)
                  + f"   离目标 {err:5.2f}°   {st}".ljust(40), end="", flush=True)
            if err < 0.3:
                break
            if time.time() - t0 > 3.0 and str(st).endswith("idle"):
                print("\n  ⚠ 控制器已 idle 但没到目标 —— 还是被拒了")
                break
            if time.time() - t0 > 4.0 and abs(qd[bad[0]] - np.rad2deg(q[bad[0]])) < 0.05:
                print("\n  ⚠ 4 秒了关节一点没动 —— 控制器接了指令但不执行")
                break
            r, _, _ = select.select([sys.stdin], [], [], 0.2)
            if r:
                sys.stdin.readline()
                print("\n  手动停止")
                break
        print()
        ev = robot.queryEventInfo(sdk.Event.moveExecution, {})
        print(f"  moveExecution: reachTarget={ev.get('reachTarget')}  {ev.get('error')}")
    except KeyboardInterrupt:
        print("\n  Ctrl+C")
    finally:
        if moved:
            # 点动后**必须** stop()（官方 Note）；刚停下时可能报 -18（状态不允许），稍等重试
            for k in range(5):
                if step(f"stop()（第 {k + 1} 次）", robot.stop) == 0:
                    break
                time.sleep(0.4)
        if limits_off:
            # 收尾固定走实测能成功的顺序：手动 → 下电 → 开软限位（不改数值）→ 自动。
            # 改软限位在自动模式下是 -16，所以不再先试自动模式。
            step("手动模式", robot.setOperateMode, sdk.OperateMode.manual)
            step("下电", robot.setPowerState, False)
            if step("**重新打开软限位**（不改数值）", robot.setSoftLimit, True) != 0:
                time.sleep(0.5)
                step("**重新打开软限位**（重试）", robot.setSoftLimit, True)
            step("自动模式", robot.setOperateMode, sdk.OperateMode.automatic)
            en, lims = soft_limits()
            same = np.allclose(lims, lims0, atol=1e-6)
            if en and same:
                print("  ✓ 软限位已恢复：开启，各轴数值与关之前逐一相同")
            else:
                print("\n" + "!" * 66)
                print(f"!! 软限位没有恢复好：开启={en}  数值一致={same}")
                print("!! **在恢复之前不要跑任何遥操**。把这段输出发出来。")
                print("!" * 66)
        q = np.rad2deg(joints_rad())
        print("  关节角 " + " ".join(f"j{i}:{v:+.1f}°" for i, v in enumerate(q)))
        print(f"  电源 {robot.powerState({})}   模式 {robot.operateMode({})}   运动状态 {robot.operationState({})}")
        robot.disconnectFromRobot({})

    still_bad = [i for i in range(7) if not (lims0[i][0] <= np.deg2rad(q[i]) <= lims0[i][1])]
    if still_bad:
        print("\n  ✗ 还有关节在限位外: " + ", ".join(f"j{i}" for i in still_bad))
        return 1
    print("\n  ✓ 都回到限位内了。下一步：python home_arm.py（看「回位后」那行，别只看「完成」）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
