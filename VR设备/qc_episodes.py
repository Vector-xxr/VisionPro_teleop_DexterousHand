#!/usr/bin/env python3
"""采完一批跑一遍，把能自动查的都查掉。只读文件，不碰硬件。

    python qc_episodes.py
    python qc_episodes.py --src ~/atom_episodes --verbose

查十项：丢帧 / 重复帧 / 画面丢失 / 帧率 / 卡顿 / 相机漂移 / 人手误入 /
有效动作 / 末端行程 / 场景多样性。外加成功标记和任务描述的统计。

**有两件它查不出来，只能自己回放看**：
  * 人手误入 —— 肤色阈值会和木桌、黄胶带、肉色物体撞色，报出来的多半是误报
  * 操作对不对 —— 抓稳没有、放到位没有，这个没有自动判据

    python replay_episode.py --all --only-flagged
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

# state/action 前 7 维是关节角 rad，后面是末端执行器。
# 末端是夹爪(1 维 0~1)还是因时手(6 维 0~1000)，从每条数据自己读，不写死。


# ── 扫 ────────────────────────────────────────────────────────────────────
def scan(files, names, verbose=False):
    import cv2
    out = []
    from episode_recorder import read_layout, LayoutError
    skipped = []
    for k, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        try:
            arm, dof, names_ee, (lo, hi), conn = read_layout(d)
        except LayoutError as e:
            skipped.append((os.path.basename(f), str(e)))
            continue
        rec = {
            "name": os.path.basename(f)[:-4],
            "n": len(d["state"]),
            "arm": arm, "dof": dof, "ee_names": names_ee,
            "lo": lo, "hi": hi, "conn": conn,
            "state": d["state"], "action": d["action"], "ee": d["ee_pose"],
            "fps": float(d["fps"]),
            "fps_measured": float(d["fps_measured"]) if "fps_measured" in d else None,
            "stamps": d["timestamp"] if "timestamp" in d else None,
            "task": str(d["task"]),
            "success": bool(d["success"]) if "success" in d else None,
            "cams": {},
        }
        for cam in names:
            cap = cv2.VideoCapture(f"{f[:-4]}_{cam}.mp4")
            diffs, bright, std, skin, first, prev = [], [], [], [], None, None
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                s = cv2.resize(fr, (160, 120))
                g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if first is None:
                    first = g
                if prev is not None:
                    diffs.append(float(np.abs(g - prev).mean()))
                prev = g
                bright.append(float(g.mean()))
                std.append(float(g.std()))
                hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
                skin.append(float((cv2.inRange(hsv, (0, 40, 60), (25, 180, 255)) > 0).mean()))
            cap.release()
            rec["cams"][cam] = {"n": len(bright), "diffs": np.array(diffs, np.float32),
                                "bright": np.array(bright, np.float32),
                                "std": np.array(std, np.float32),
                                "skin": np.array(skin, np.float32), "first": first}
        out.append(rec)
        if verbose:
            print(f"\r  扫描 {k+1}/{len(files)}", end="", flush=True)
    if verbose:
        print()
    for name, why in skipped:
        print(f"  ✗ 跳过 {name}：{why}")
    if skipped:
        print(f"  共跳过 {len(skipped)} 条，它们不参与下面任何一项检查")
    return out


# ── 判 ────────────────────────────────────────────────────────────────────
def check(recs, names, thresh_px=8.0):
    import cv2
    problems, n_eps = [], len(recs)
    print(f"\n  {n_eps} 条，共 {sum(r['n'] for r in recs)} 帧\n")

    # 1 丢帧
    bad = [r["name"] for r in recs if any(c["n"] != r["n"] for c in r["cams"].values())]
    print(f"  1 丢帧        npz 与各路 mp4 帧数：{'✓ 全部一致' if not bad else f'✗ {bad}'}")
    if bad:
        problems.append(f"帧数对不上：{bad}")
    L = np.array([r["n"] for r in recs])
    hz = np.median([r["fps_measured"] or r["fps"] for r in recs])
    print(f"                帧数 {L.min()}~{L.max()}  中位 {int(np.median(L))}"
          f"  = {np.median(L)/max(hz,1e-6):.1f} 秒")

    # 2 重复帧：帧间差恰好为 0 = 同一张图被读了两次
    for cam in names:
        z = sum(int((r["cams"][cam]["diffs"] == 0).sum()) for r in recs)
        t = sum(len(r["cams"][cam]["diffs"]) for r in recs)
        print(f"  2 重复帧      {cam:10} {'✓' if z == 0 else f'✗ {z}'} / {t} 帧")
        if z:
            problems.append(f"{cam} 有 {z} 帧重复")

    # 3 画面丢失
    for cam in names:
        blk = sum(int(((r["cams"][cam]["bright"] < 10) |
                       (r["cams"][cam]["std"] < 3)).sum()) for r in recs)
        b = np.concatenate([r["cams"][cam]["bright"] for r in recs])
        print(f"  3 画面丢失    {cam:10} {'✓ 0 黑帧' if not blk else f'✗ {blk} 黑帧'}"
              f"   亮度 {b.min():.0f}~{b.max():.0f}")
        if blk:
            problems.append(f"{cam} 有 {blk} 个黑帧")

    # 4 帧率
    have = [r for r in recs if r["fps_measured"]]
    if have:
        m = np.array([r["fps_measured"] for r in have])
        nominal = have[0]["fps"]
        print(f"  4 帧率        实测 {m.min():.2f}~{m.max():.2f} Hz  中位 {np.median(m):.2f}"
              f"   标称 {nominal:.0f}  达成 {np.median(m)/nominal*100:.1f}%")
        slow = [r["name"] for r in have if r["fps_measured"] < nominal * 0.9]
        if slow:
            problems.append(f"帧率低于标称 90%：{slow}")
        stamps = [np.diff(r["stamps"]) * 1000 for r in recs if r["stamps"] is not None]
        if stamps:
            dt = np.concatenate(stamps)
            gap = 3000.0 / max(nominal, 1e-6)     # 超过三个采样周期 = 录制线卡了一下
            over = int((dt > gap).sum())
            print(f"                逐帧间隔 p50 {np.median(dt):.1f} p99 "
                  f"{np.percentile(dt,99):.1f} 最大 {dt.max():.0f} ms   >{gap:.0f}ms 的 {over} 处")
            if over:
                problems.append(f"有 {over} 处帧间隔超过 {gap:.0f} ms —— 录制线程被挡过")

    # 5 卡顿
    for cam in names:
        hit = 0
        for r in recs:
            c = 0
            for v in (r["cams"][cam]["diffs"] < 0.1):
                c = c + 1 if v else 0
                if c >= 5:
                    hit += 1
                    break
        print(f"  5 卡顿        {cam:10} 连续≥5 帧几乎不变的 episode "
              f"{'✓ 0' if not hit else f'✗ {hit}'} / {n_eps}")
        if hit:
            problems.append(f"{cam} 有 {hit} 条卡顿")

    # 6 相机漂移
    #
    # **只在「跨 episode 不变的像素」上量。** 整帧比对会被两样东西带偏：臂在首帧
    # 的位置（每条的起始位姿本来就有几度的自然离散）和物体的摆位。静态区不硬编码
    # 坐标，用跨所有首帧逐像素标准差最低的四成自己算出来，换机位也不用改。
    print(f"  6 相机漂移    只在跨 episode 不变的像素上量（臂和物体的位置会带偏整帧）")
    for cam in names:
        firsts = [r["cams"][cam]["first"] for r in recs if r["cams"][cam]["first"] is not None]
        if len(firsts) < 5:
            print(f"                {cam:10} 只有 {len(firsts)} 条，样本太少，不量")
            continue
        stack = np.array(firsts, np.float32)
        static = stack.std(0) < np.percentile(stack.std(0), 40)
        bg, shifts = np.median(stack, 0), []
        for r in recs:
            f = r["cams"][cam]["first"]
            if f is None:
                continue
            (dx, dy), _ = cv2.phaseCorrelate(np.where(static, bg, 0).astype(np.float64),
                                             np.where(static, f, 0).astype(np.float64))
            shifts.append((r["name"], np.hypot(dx, dy) * 4))     # 160x120 -> 640x480
        mag = np.array([s[1] for s in shifts])
        worst = max(shifts, key=lambda s: s[1])
        print(f"                {cam:10} 相对静态背景 中位 {np.median(mag):.1f} "
              f"最大 {mag.max():.1f} px  （静态像素 {100*static.mean():.0f}%）"
              f"  {'✓ 没动' if mag.max() < thresh_px else f'⚠ 最大在 {worst[0]}'}")
        if mag.max() >= thresh_px:
            problems.append(f"{cam} 首帧最大漂移 {mag.max():.0f} px（{worst[0]}）")

    # 7 人手误入
    for cam in names:
        mx = np.array([r["cams"][cam]["skin"].max() for r in recs])
        top = sorted(zip([r["name"] for r in recs], mx), key=lambda x: -x[1])[:3]
        print(f"  7 人手误入    {cam:10} 肤色峰值 中位 {np.median(mx)*100:.1f}% "
              f"最大 {mx.max()*100:.1f}%   最高三条 {[f'{n}:{v*100:.0f}%' for n, v in top]}")
    print(f"                ⚠ 这个判据不可靠 —— 木桌、黄胶带都会撞色。**要靠回放确认**")

    # 8 有效动作
    #
    # 动作空间是**绝对关节角**，不是增量，所以「这一帧动没动」要从末端位姿的
    # 逐帧差算，不能拿 action 的模长当位移。
    lead, tail, mid, tot, useful = [], [], [], [], []
    for r in recs:
        p, h = r["ee"][:, :3], r["action"][:, r["arm"]:]
        step = np.concatenate([[0.0], np.linalg.norm(np.diff(p, axis=0), axis=1) * 1000])
        # 末端"这一帧动没动"的阈值按量程的 0.2% 定，夹爪(0~1)和手(0~1000)同一条规则
        eps = (r["hi"] - r["lo"]) * 0.002
        dh = (np.abs(np.diff(h, axis=0, prepend=h[:1])).max(1) if r["dof"]
              else np.zeros(len(p)))
        act = (step > 0.4) | (dh > eps)
        k = np.convolve(act.astype(float), np.ones(5) / 5, mode="same") > 0.4
        n = len(p)
        if not k.any():
            lead.append(n); tail.append(0); mid.append(0); tot.append(n); useful.append(0)
            continue
        first, last = int(np.argmax(k)), n - 1 - int(np.argmax(k[::-1]))
        lead.append(first); tail.append(n - 1 - last)
        mid.append(int((~k[first:last + 1]).sum())); tot.append(n); useful.append(int(k.sum()))
    lead, tail, mid, tot, useful = map(np.array, (lead, tail, mid, tot, useful))
    print(f"  8 有效动作    {useful.sum()/tot.sum()*100:.1f}%"
          f"   开头空转 {lead.sum()/tot.sum()*100:.1f}%"
          f"  结尾 {tail.sum()/tot.sum()*100:.1f}%"
          f"  中间停顿 {mid.sum()/tot.sum()*100:.1f}%")
    if lead.sum() / tot.sum() > 0.15:
        problems.append(f"开头有 {lead.sum()/tot.sum()*100:.0f}% 是空转 —— "
                        f"按空格按早了，或者 reset 还没做完就开录")

    # 9 末端的行程
    #
    # **不要拿「有没有用满量程」当判据。** 任务用不到量程顶部不是缺陷，归一化会
    # 自己适配。真正要查的是三件：每条内部张合的行程够不够、有没有真正合拢过、
    # 有没有哪条全程没动。阈值都按量程的百分比定，夹爪和因时手同一条规则。
    dofs = {r["dof"] for r in recs}
    if not dofs - {0}:
        print(f"  9 末端行程    这批数据没有末端维度，跳过")
    else:
        if len(dofs) > 1:
            problems.append(f"同一目录里混了不同的末端维度 {sorted(dofs)} —— "
                            f"换过硬件？分开放，别混在一个数据集里")
        rng = recs[0]["hi"] - recs[0]["lo"]
        kind = "夹爪" if recs[0]["dof"] == 1 else f"末端{recs[0]['dof']}维"
        span = np.array([float(np.max(r["action"][:, r["arm"]:].max(0)
                                      - r["action"][:, r["arm"]:].min(0)))
                         for r in recs if r["dof"]])
        allh = np.concatenate([r["action"][:, r["arm"]:] for r in recs if r["dof"]])
        closed = float((allh.min(1) < recs[0]["lo"] + rng * 0.08).mean())
        print(f"  9 末端行程    {kind} 逐条最大行程 中位 {np.median(span):.3g}"
              f"（{span.min():.3g} ~ {span.max():.3g}，满量程 {rng:.3g}）"
              f"   合拢到底的帧 {closed*100:.0f}%")
        if np.median(span) < rng * 0.15:
            problems.append(f"末端逐条行程只有满量程的 {np.median(span)/rng*100:.0f}%，几乎没张合过")
        if closed < 0.03:
            problems.append(f"只有 {closed*100:.1f}% 的帧末端合拢到底 —— 抓取可能没真正夹住")
        dead = [r["name"] for r in recs if r["dof"] and
                float(np.max(r["action"][:, r["arm"]:].max(0)
                             - r["action"][:, r["arm"]:].min(0))) < rng * 0.05]
        if dead:
            problems.append(f"这几条末端全程没动：{dead[:8]}{' ...' if len(dead) > 8 else ''}")
        # 末端没接线时 state 的末端维是拿指令顶的，不是真实测值
        noconn = [r["name"] for r in recs if r["dof"] and not r["conn"]]
        if noconn:
            print(f"                ⚠ {len(noconn)}/{len(recs)} 条的末端**没有位置反馈**，"
                  f"state 那几维等于 action。训练时别把它当独立观测")

    # 10 场景多样性：固定场景训不出泛化
    S = np.array([r["ee"][0, :3] for r in recs]) * 1000
    print(f" 10 多样性      起始末端 std (x,y,z) = "
          f"({S[:,0].std():.0f}, {S[:,1].std():.0f}, {S[:,2].std():.0f}) mm"
          f"   跨度 ({np.ptp(S[:,0]):.0f}, {np.ptp(S[:,1]):.0f}, {np.ptp(S[:,2]):.0f}) mm")

    # 成功标记与任务描述
    marked = [r["success"] for r in recs if r["success"] is not None]
    if marked:
        ok = sum(marked)
        print(f"    成功标记    {ok}/{len(marked)} 条按 s 存的"
              f"（其余按 ← 保留，训练时按 success 字段筛）")
    tasks = {r["task"] for r in recs}
    if len(tasks) > 1:
        problems.append(f"任务描述不一致：{tasks}")
    if tasks == {""}:
        problems.append("任务描述是空的 —— 采的时候没给 --task，语言条件对不上")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="~/atom_episodes")
    ap.add_argument("--thresh", type=float, default=8.0, help="相机漂移阈值 px")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    src = os.path.expanduser(a.src)
    files = sorted(glob.glob(f"{src}/episode_*.npz"))
    if not files:
        print(f"  {src} 里没有数据")
        return 1

    names = [str(k) for k in np.load(files[0], allow_pickle=True)["camera_keys"]]
    print(f"  {src}   相机 {names}")
    recs = scan(files, names, verbose=True)
    if not recs:
        print("\n  一条都读不了 —— 这个目录里没有本流水线格式的数据")
        return 1
    problems = check(recs, names, a.thresh)

    print()
    if problems:
        print(f"  ── 要处理的 {len(problems)} 项 ──")
        for p in problems:
            print(f"    ⚠ {p}")
    else:
        print(f"  ── 自动检查全过 ──")
    print(f"\n  剩下这两件只能回放看：人手误入、操作对不对")
    print(f"    python replay_episode.py --src {a.src} --all --only-flagged")
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
