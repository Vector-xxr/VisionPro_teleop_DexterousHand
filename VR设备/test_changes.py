#!/usr/bin/env python3
"""离线自检：握拳耦合规则 + 新逆解对驱动的契约。**不需要任何硬件。**

改完 inspire_hand6.clamp 或 ar5_ik.AR5OptIK 之后先跑这个，全绿再上真机。

    python test_changes.py
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ws_paths import VEL_AR5  # noqa: E402
sys.path.insert(0, str(VEL_AR5))

from inspire_hand6 import INDEX, InspireHand6, index_floor  # noqa: E402

FAILED = []


def ck(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗ **FAIL**'} {name}  {detail}")
    if not cond:
        FAILED.append(name)


# ════════════════════════════════════════════════════════════════════════
def test_hand():
    """食指耦合下限：拦「压下去」，放「张开」，永远不把食指往外推。"""
    print("\n【手】握拳时拇指把食指顶开 —— 就是这条在回归")

    def hand(couple=True):
        # serial_factory 返回 None：只测 clamp，不碰串口
        return InspireHand6(serial_factory=lambda: None, couple_index_thumb=couple)

    # 1) 复现用户报的动作：先卷四指（食指到 0），拇指后到
    for couple in (True, False):
        h = hand(couple)
        for t in range(1000, -1, -100):
            h.clamp([t] * 4 + [1000, 100])          # 四指含食指卷到 0
        curl = [h.clamp([0, 0, 0, 0, tb, 100])[INDEX]
                for tb in range(1000, -1, -100)]    # 拇指随后收拢
        ck(f"握拳 couple={couple}: 收拇指全程食指不动", max(curl) == 0,
           f"食指取值 {sorted(set(curl))}")

    # 2) 捏合仍然要拦得住（这条下限本来就是为捏合量的）
    h = hand()
    got = [h.clamp([1000] * 3 + [v, 500, 100])[INDEX] for v in range(1000, -1, -100)]
    ck("捏合(拇指=500): 食指压下去被拦在下限",
       min(got) == index_floor(500), f"最低 {min(got)}, 下限 {index_floor(500)}")

    # 3) 离开拇指的方向永远安全，必须放行
    h = hand()
    for v in range(1000, -1, -100):
        h.clamp([1000] * 3 + [v, 500, 100])         # 先压到下限
    back = [h.clamp([1000] * 3 + [v, 500, 100])[INDEX] for v in (500, 700, 1000)]
    ck("捏住后张开食指: 放行", back == [500, 700, 1000], f"{back}")

    # 4) 拇指让开后食指要能全闭（否则又回到「单独弯食指没反应」）
    h = hand()
    for v in range(1000, -1, -100):
        h.clamp([1000] * 3 + [v, 500, 100])
    ck("拇指让开(800)后食指能全闭", h.clamp([1000] * 3 + [0, 800, 100])[INDEX] == 0)

    # 5) 第一帧没有历史，不许开局就把食指弹出去
    ck("首帧「食指0/拇指0」不推", hand().clamp([0, 0, 0, 0, 0, 100])[INDEX] == 0)

    # 6) 关掉耦合时退化成纯静态限位
    ck("couple=False 只做静态限位",
       hand(False).clamp([-5, 0, 500, 0, 0, 2000]) == [0, 0, 500, 0, 0, 400])


# ════════════════════════════════════════════════════════════════════════
DH = [(0, 0, 342, 0), (-90, 0, 0, 0), (90, 0, 400, 0), (90, 0, 0, 0),
      (-90, 0, 400, 0), (-90, 0, 0, 0), (90, 0, 126, 0)]
JMIN = (-3.1067, -2.0944, -3.1067, -1.0472, -3.1067, -0.8727, -0.8727)
JMAX = (3.1067, 2.0944, 3.1067, 2.5307, 3.1067, 0.8727, 0.8727)
HOME = np.array([0.006413, 0.330561, -0.017369, 1.926388, -0.831742,
                 0.066788, -0.143514])
MARGIN, JUMP_MAX = 0.05, 0.15       # config.joint_limit_margin / joint_jump_max


def aa2m(v):
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def test_ik():
    """新逆解交出去的每个解，都必须过得了驱动的四道闸。"""
    from vel_kin import AR5Kinematics
    from ar5_ik import AR5OptIK

    print("\n【臂】AR5OptIK 对 rokae_arm 的契约")
    kin = AR5Kinematics(DH, 7, JMIN, JMAX)
    s = AR5OptIK(kin, limit_margin=MARGIN, max_step=0.12)
    lo, hi = np.array(JMIN) + MARGIN, np.array(JMAX) - MARGIN

    # 驱动的调用形态：kinematics.ik(T, seed)，两个位置参数
    T = kin.fk(HOME).copy()
    T[:3, 3] += [0.03, 0.02, 0.01]
    q, info = s.ik(T, HOME)
    ck("ik(T, seed) 返回 (q, info)", q is not None and isinstance(info, dict))
    ck("info 带 'reason'（驱动失败时会打印）", "reason" in info)

    # 梯度必须和数值差分一致，否则 L-BFGS-B 的线搜索会乱走
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(100):
        qq = np.clip(HOME + rng.normal(0, 0.4, 7), lo, hi)
        Tt = kin.fk(np.clip(HOME + rng.normal(0, 0.5, 7), lo, hi))
        _, g = s._cost_grad(qq, Tt, HOME)
        gn = np.array([
            (s._cost_grad(np.where(np.arange(7) == i, qq + 1e-6, qq), Tt, HOME)[0]
             - s._cost_grad(np.where(np.arange(7) == i, qq - 1e-6, qq), Tt, HOME)[0]) / 2e-6
            for i in range(7)])
        worst = max(worst, float(np.linalg.norm(g - gn) / max(np.linalg.norm(gn), 1e-9)))
    ck("解析梯度 == 数值差分", worst < 1e-5, f"最大相对误差 {worst:.2e}")

    # 遥操里真实会出现的目标：两条缰绳之内（平移 4cm、姿态 0.35rad）
    rng = np.random.default_rng(1)
    step_worst, solved, out_lim = 0.0, 0, 0
    for _ in range(400):
        seed = np.clip(HOME + rng.normal(0, 0.3, 7), lo, hi)
        Tn = kin.fk(seed)
        Tt = np.eye(4)
        u = rng.normal(0, 1, 3); u /= np.linalg.norm(u)
        w = rng.normal(0, 1, 3); w /= np.linalg.norm(w)
        Tt[:3, 3] = Tn[:3, 3] + u * 0.04
        Tt[:3, :3] = aa2m(w * 0.35) @ Tn[:3, :3]
        qq, _ = s.ik(Tt, seed)
        if qq is None:
            continue
        solved += 1
        step_worst = max(step_worst, float(np.max(np.abs(qq - seed))))
        if np.any(qq < lo - 1e-9) or np.any(qq > hi + 1e-9):
            out_lim += 1
    ck("测试非空（有解才谈得上守约束）", solved > 300, f"给解 {solved}/400")
    ck("闸3 关节限位: 解全在软限位内", out_lim == 0, f"越限 {out_lim} 个")
    ck(f"闸4 跳变: 单关节步长 ≤ 0.12 < {JUMP_MAX}", step_worst <= 0.12 + 1e-9,
       f"实测最大 {step_worst:.5f} rad")

    # 透传：驱动和脚本还会用 base 上的这些
    for a in ("fk", "jacobian", "pose_error", "joint_min", "joint_max",
              "num_joints", "limit_report", "near_singularity", "manipulability"):
        if not hasattr(s, a):
            ck(f"透传 {a}", False)
    ck("属性透传齐全 + fk 与原对象一致", np.allclose(s.fk(HOME), kin.fk(HOME)))


def test_imports():
    """回归：环境里有个同名 PyPI 包 `kinematics`，会把 vel_ar5 的顶掉。

    实机上报出来是 `ModuleNotFoundError: No module named 'kinematics'`
    （或者更隐蔽的 `cannot import name 'AR5Kinematics'`）——取决于调用方
    有没有恰好把 vel_ar5 插在 sys.path 前面。按文件路径加载才是确定的。
    """
    print("\n【导入】同名 PyPI 包不许把 vel_ar5 的运动学顶掉")
    import importlib
    from vel_kin import load_vel_module
    src = load_vel_module("kinematics").__file__
    ck("拿到的是 vel_ar5 的 kinematics.py", "vel_ar5" in src, src)
    # 只做提示，不做断言：`import kinematics` 拿到谁本来就随 sys.path 变，
    # 那正是这里要消除的不确定性 —— 拿它当断言等于把不确定性写进测试。
    try:
        import kinematics as amb
        where = "site-packages（同名 PyPI 包）" if "site-packages" in (amb.__file__ or "") \
                else "vel_ar5"
        print(f"    （参考）裸 `import kinematics` 此刻解析到: {where}")
    except ImportError:
        print("    （参考）裸 `import kinematics` 此刻解析不到任何东西")
    # 每个求解器模块都必须能独立导入，不靠调用方铺路
    for mod in ("ar5_ik", "ar5_srs_ik", "ar5_poe_ik", "paden_kahan", "vel_kin"):
        try:
            importlib.import_module(mod)
            ok = True
        except Exception as exc:                       # noqa: BLE001
            ok = False
            print(f"    {mod}: {type(exc).__name__}: {exc}")
        ck(f"{mod} 可独立导入", ok)


def test_movel_buffer():
    """本地缓存 + 分频下发。**不碰硬件**：robot/sdk 都是假的，只记调用。

    回归的是四条容易写错的性质：
      · 死区锚点跨下发周期累积；小于阈值的剩余误差不会强制下发
      · 超预算时分步追踪，最新终点留在本地供下一周期使用
      · 一批的代价是**折线长度**不是直线距离（队列语义：每个点都得走完）
      · flush 返回的是**最后真正下发**那个点的位姿（拿它同步 _target_pose）
    """
    print("\n【MoveL 缓存】100Hz 进缓存 / 20Hz 批量下发")
    from movel_stream import MoveLStreamer

    class FakeSdk:
        class CoordinateType:
            flangeInBase = "f"

        class OperationState:
            moving = "moving"
            idle = "idle"

        class CartesianPosition:
            def __init__(self):
                self.trans, self.rpy = [0.0] * 3, [0.0] * 3
                self.elbow, self.hasElbow, self.confData = 0.0, False, [0] * 8

        class MoveLCommand:
            def __init__(self, t):
                self.target, self.speed, self.zone = t, 0.0, 0.0

        @staticmethod
        def PyString(s):
            return s

    class FakeRobot:
        """臂停在原点不动，状态恒为 moving（不触发 moveStart 那条路）。"""

        def __init__(self):
            self.appended, self.state, self.calls = [], "moving", []
            self.tcp = np.zeros(3)

        def cartPosture(self, _c, ec):
            p = FakeSdk.CartesianPosition()
            p.trans = list(self.tcp)
            return p

        def operationState(self, ec):
            return self.state

        def moveAppend(self, cmd, name, ec):
            # 两种形态都收，和真 SDK 的重载对齐（实测 R.moveAppend.__doc__）：
            #   #2 moveAppend(cmds: list[MoveLCommand], cmdID, ec)
            #   #8 moveAppend(cmd:  MoveLCommand,       cmdID, ec)
            # 夹具只认单条的话，「改走批量」这个改动会在测试里表现成假的失败。
            for c in (cmd if isinstance(cmd, (list, tuple)) else [cmd]):
                self.appended.append(np.asarray(c.target.trans, dtype=float))
            self.calls.append(cmd)

        def moveStart(self, ec):
            ec["ec"] = -20                      # 良性：机器人运动中

        def moveReset(self, ec):
            pass

        def setMotionControlMode(self, *a):
            pass

    def mk(deadband=3.0, lead_ms=50.0, speed=250.0, buffer_max=64):
        r = FakeRobot()
        s = MoveLStreamer(r, FakeSdk, speed=speed, lead_ms=lead_ms,
                          min_step_mm=deadband, buffer_max=buffer_max)
        s._running = True
        s._elbow, s._conf, s._last_rpy = 0.0, None, [0.0] * 3
        s._last_target = np.zeros(3)
        return r, s

    # 1) 采样不碰SDK；死区在flush生效，慢速位移跨采样累积
    r, s = mk(deadband=3.0)
    for i in range(1, 13):                       # 每帧 0.75mm，12 帧 = 9mm
        s.submit(np.array([i * 0.00075, 0.0, 0.0]))
    ck("submit 不碰 SDK", not r.appended, f"append {len(r.appended)} 次")
    ck("所有新采样进入本地缓存", s.stats["buffered"] == 12)
    s.flush()
    ck("慢速位移累计过阈后下发", len(r.appended) == 3 and np.allclose(r.appended[-1], [.009,0,0]))

    # 2) 预算够 → 整批原样发出，顺序不变 ---------------------------------
    r, s = mk(deadband=3.0, lead_ms=200.0)       # 预算 50mm
    pts = [np.array([i * 0.004, 0.0, 0.0]) for i in range(1, 5)]   # 4 点，折线 16mm
    for p_ in pts:
        s.submit(p_, aa=np.zeros(3))
    res, last6 = s.flush()
    ck("预算够时整批发出", len(r.appended) == 4, f"发了 {len(r.appended)} 点，{res}")
    ck("顺序没乱", all(np.allclose(a, b) for a, b in zip(r.appended, pts)))
    ck("flush 返回最后下发点", last6 is not None and np.allclose(last6[:3], pts[-1]),
       f"last6={None if last6 is None else np.round(last6[:3], 4).tolist()}")

    # 3) 预算不够 → 抽稀，但最新点一定在 ---------------------------------
    r, s = mk(deadband=3.0, lead_ms=20.0)        # 预算 5mm
    pts = [np.array([i * 0.004, 0.0, 0.0]) for i in range(1, 6)]   # 5 点，折线 20mm
    for p_ in pts:
        s.submit(p_, aa=np.zeros(3))
    res, last6 = s.flush()
    ck("超预算会抽稀", 0 < len(r.appended) < 5, f"发了 {len(r.appended)}/5 点")
    ck("超预算目标分步前进且保留最终目标", np.allclose(r.appended[-1], [.005,0,0])
       and s.pending_points() == 1 and np.allclose(s._buf[-1][0], pts[-1]))
    _plen = s._path_len([(p_, None, None) for p_ in r.appended], np.zeros(3))
    ck("抽稀后：装进预算，或已抽到只剩最新一点",
       _plen <= s.max_lead + 1e-9 or len(r.appended) == 1,
       f"折线 {_plen*1000:.1f}mm vs 预算 {s.max_lead*1000:.1f}mm，"
       f"发了 {len(r.appended)} 点，单点超预算计数 {s.stats['over_budget']}")
    ck("单点超预算要记账（否则现场看不见预算太紧）",
       s.stats["over_budget"] == 1, f"over_budget={s.stats['over_budget']}")

    # 3b) **抽稀只对带抖动的路径有用** ------------------------------------
    # 直线运动上抽稀一点用都没有：任何以最新点收尾的子集，折线长度都 ≥ 起点到
    # 最新点的**直线距离**（三角不等式），砍中间点砍不出预算来。
    # 折线远长于直线时（=路径在抖）才回收得到 —— 这正是死区那套道理的同一面。
    r, s = mk(deadband=0.5, lead_ms=40.0)        # 预算 10mm
    zig = [0.004, 0.001, 0.005, 0.002, 0.006]    # 来回拉锯：折线 18mm，直线只 6mm
    pts = [np.array([x, 0.0, 0.0]) for x in zig]
    for p_ in pts:
        s.submit(p_, aa=np.zeros(3))
    ck("锯齿路径的折线确实远长于直线",
       abs(s._path_len([(p_, None, None) for p_ in pts], np.zeros(3)) - 0.018) < 1e-9,
       f"折线 {s._path_len([(p_, None, None) for p_ in pts], np.zeros(3))*1000:.1f}mm "
       f"vs 直线 {zig[-1]*1000:.1f}mm")
    s.flush()
    _plen = s._path_len([(p_, None, None) for p_ in r.appended], np.zeros(3))
    ck("抖动路径上抽稀能把折线砍进预算",
       2 <= len(r.appended) < 5 and _plen <= s.max_lead + 1e-9,
       f"发了 {len(r.appended)}/5 点，折线 {_plen*1000:.1f}mm ≤ 预算 "
       f"{s.max_lead*1000:.1f}mm")
    ck("抽稀保留的仍以最新点收尾", np.allclose(r.appended[-1], pts[-1]))
    ck("抽稀没白丢（真省出了路径）", _plen < 0.018 - 1e-9,
       f"{_plen*1000:.1f}mm < 原折线 18.0mm")

    # 4) 前导已吃光预算且臂在动 → 整批丢，不加深积压 ----------------------
    r, s = mk(deadband=3.0, lead_ms=20.0)
    s._last_target = np.array([0.050, 0.0, 0.0])     # 已领先实测 50mm >> 预算 5mm
    for i in range(1, 4):
        s.submit(np.array([0.050 + i * 0.004, 0.0, 0.0]), aa=np.zeros(3))
    res, last6 = s.flush()
    ck("超前导整批丢", not r.appended and res == "lead",
       f"发了 {len(r.appended)} 点，{res}")
    ck("丢的计进 skipped_lead", s.stats["skipped_lead"] == 2 and s.pending_points() == 1,
       f"skipped_lead={s.stats['skipped_lead']}")

    # 5) 臂停了就算超预算也必须发，否则没人再启动它 ------------------------
    r, s = mk(deadband=3.0, lead_ms=20.0)
    r.state = "idle"
    s._last_target = np.array([0.050, 0.0, 0.0])
    s.submit(np.array([0.070, 0.0, 0.0]), aa=np.zeros(3))
    res, last6 = s.flush()
    ck("臂停了不许空过", len(r.appended) == 1, f"发了 {len(r.appended)} 点，{res}")

    # 5b) 自适应段速：手慢→段速降，手快→顶到上限 --------------------------
    r, s = mk(deadband=3.0, speed=250.0)
    ck("不开自适应时段速恒为 --movel-speed",
       s.segment_speed() == 250.0, f"{s.segment_speed()}")
    s.adaptive_speed, s.min_speed = True, 20.0
    s._v_hand = 0.0
    ck("手没动时落到下限（不能给 0，否则段永远走不完）",
       s.segment_speed() == 20.0, f"{s.segment_speed()}")
    s._v_hand = 30.0                             # 慢速：臂短段实测就是 25~33mm/s
    ck("手慢时段速跟着降（这才是不跑空的关键）",
       abs(s.segment_speed() - 39.0) < 1e-6, f"手30mm/s → 段 {s.segment_speed():.0f}mm/s")
    s._v_hand = 400.0
    ck("手快时顶到上限，不超过 --movel-speed",
       s.segment_speed() == 250.0, f"{s.segment_speed()}")

    # 5c) 段速真的写进了 MoveLCommand（不是算了不用）-----------------------
    r, s = mk(deadband=3.0, speed=250.0, lead_ms=200.0)
    s.adaptive_speed, s.min_speed, s._v_hand = True, 20.0, 40.0
    _seen = []
    _orig = r.moveAppend
    r.moveAppend = lambda cmd, n, ec, _o=_orig: (_seen.extend(c.speed for c in (cmd if isinstance(cmd, list) else [cmd])), _o(cmd, n, ec))[1]
    s.submit(np.array([0.004, 0.0, 0.0]), aa=np.zeros(3))
    s.flush()
    ck("段速写进了 MoveLCommand.speed",
       _seen and abs(_seen[0] - 52.0) < 1e-6,
       f"下发时 cmd.speed={_seen[0] if _seen else None}（手40×1.3=52）")

    # 5e) 批量 moveAppend：一次调用提交整批 ---------------------------------
    # SDK 实测有两组重载：#2 moveAppend(list[MoveLCommand],...) / #8 单条。
    # 原来走 #8 逐条，改走 #2。
    r, s = mk(deadband=3.0, lead_ms=400.0)       # 预算 100mm，够整批发
    _calls = []
    r.moveAppend = lambda cmd, n, ec, _r=r: (
        _calls.append(cmd),
        [_r.appended.append(np.asarray(c.target.trans, dtype=float))
         for c in (cmd if isinstance(cmd, list) else [cmd])])[0]
    s.batch_append = True
    pts = [np.array([i * 0.004, 0.0, 0.0]) for i in range(1, 5)]
    for p_ in pts:
        s.submit(p_, aa=np.zeros(3))
    sent, last6 = s.flush()[0], None
    ck("批量：4 个点只调 1 次 moveAppend", len(_calls) == 1,
       f"调了 {len(_calls)} 次")
    ck("批量：传进去的是 list", isinstance(_calls[0], list) and len(_calls[0]) == 4,
       f"{type(_calls[0]).__name__} 长度 "
       f"{len(_calls[0]) if isinstance(_calls[0], list) else '-'}")
    ck("批量：4 个点都进了队列", len(r.appended) == 4, f"{len(r.appended)}")
    ck("批量：appended 计数按**点**算不是按调用算",
       s.stats["appended"] == 4, f"appended={s.stats['appended']}")

    # 批大小为 1 时自动退回单条形态（两者等价，不必包 list）
    r, s = mk(deadband=3.0, lead_ms=400.0)
    _calls = []
    r.moveAppend = lambda cmd, n, ec, _r=r: (
        _calls.append(cmd),
        _r.appended.append(np.asarray(
            (cmd[0] if isinstance(cmd, list) else cmd).target.trans, dtype=float)))[0]
    s.batch_append = True
    s.submit(np.array([0.004, 0.0, 0.0]), aa=np.zeros(3))
    s.flush()
    ck("批大小 1 也走列表形态（一律走 list）",
       len(_calls) == 1 and isinstance(_calls[0], list) and len(_calls[0]) == 1,
       f"{type(_calls[0]).__name__}")

    # 重载不匹配时要能退回逐条，而不是整条链路挂掉
    r, s = mk(deadband=3.0, lead_ms=400.0)
    _orig = r.moveAppend
    def _reject_list(cmd, n, ec, _o=_orig):
        if isinstance(cmd, list):
            raise TypeError("incompatible function arguments")
        return _o(cmd, n, ec)
    r.moveAppend = _reject_list
    s.batch_append = True
    for i in range(1, 4):
        s.submit(np.array([i * 0.004, 0.0, 0.0]), aa=np.zeros(3))
    s.flush()
    ck("批量重载不可用时退回逐条，不中断",
       len(r.appended) == 3 and s.batch_append is False
       and s.stats["batch_fallback"] == 1,
       f"发了 {len(r.appended)} 点，batch_append={s.batch_append}，"
       f"fallback={s.stats['batch_fallback']}")

    # 5d) 卡顿诊断：空闲占比 / 跑空次数 -------------------------------------
    r, s = mk(deadband=3.0)
    r.state = "idle"
    s._track_stall(np.zeros(3), True)
    time.sleep(0.05)
    s._track_stall(np.zeros(3), True)            # 仍空闲
    time.sleep(0.02)
    s._track_stall(np.array([0.001, 0, 0]), False)   # 动起来 → 记一次停顿
    ck("队列跑空会被记成一次停顿", len(s._stalls) == 1,
       f"stalls={[round(x*1000) for x in s._stalls]}ms")
    ck("停顿时长量得对（约 70ms）", 0.05 < s._stalls[0] < 0.12,
       f"{s._stalls[0]*1000:.0f}ms")
    ck("空闲占比算得出来", s._idle_acc > 0 and s._busy_acc > 0,
       f"空闲 {s._idle_acc*1000:.0f}ms / 忙 {s._busy_acc*1000:.0f}ms")

    # 6) 缓存溢出丢最老的 --------------------------------------------------
    r, s = mk(deadband=0.1, buffer_max=4)
    for i in range(1, 11):
        s.submit(np.array([i * 0.004, 0.0, 0.0]), aa=np.zeros(3))
    ck("溢出丢最老的", s.pending_points() == 4 and s.stats["dropped_overflow"] == 6,
       f"缓存 {s.pending_points()} 点，丢 {s.stats['dropped_overflow']}")
    ck("留下的是最新的", np.allclose(s._buf[-1][0], [0.040, 0.0, 0.0]))


def test_stuck():
    """实机回归：腕部姿态够不到时，臂被快速推进限位然后卡死。

    第一版 AR5OptIK 就是栽在这里 —— 6 帧(120ms)顶死 j5/j6，而且因为它
    永不返回 None，avp_arm_teleop 的两张自救网（打滑重锚 / 限位自救）
    **同时失效**，没有任何人吭声。用户报的「IK 直接快速超限卡住」。
    """
    from vel_kin import AR5Kinematics
    from ar5_ik import AR5OptIK

    print("\n【回归】腕部够不到时不许顶死限位，且必须报得出「跟不上」")
    kin = AR5Kinematics(DH, 7, JMIN, JMAX)
    drv_lo, drv_hi = np.array(JMIN) + MARGIN, np.array(JMAX) - MARGIN
    ML, MRL = 0.04, 0.35        # 遥操的两条缰绳

    def play(T_hand, frames=200):
        """按遥操真实管路跑：先上缰绳，再进 IK。"""
        s = AR5OptIK(kin, limit_margin=MARGIN)
        q = HOME.copy()
        for _ in range(frames):
            Tn = kin.fk(q)
            T = T_hand.copy()
            d = T_hand[:3, 3] - Tn[:3, 3]
            nd = float(np.linalg.norm(d))
            if nd > ML:
                T[:3, 3] = Tn[:3, 3] + d * (ML / nd)
            er = kin.pose_error(Tn, T_hand)[3:]
            nr = float(np.linalg.norm(er))
            if nr > MRL:
                T[:3, :3] = aa2m(er * (MRL / nr)) @ Tn[:3, :3]
            qq, _ = s.ik(T, q)
            if qq is not None:
                q = qq
        return q, s

    # 场景：人手把腕拧到 92°（j5/j6 行程只有 ±50°），然后保持不动
    T_bad = kin.fk(HOME).copy()
    T_bad[:3, :3] = aa2m(np.array([0.0, 0.0, 1.6])) @ T_bad[:3, :3]
    q, s = play(T_bad)
    pinned = [f"j{i}" for i in range(7)
              if q[i] <= drv_lo[i] + 1e-6 or q[i] >= drv_hi[i] - 1e-6]
    margin = float(np.rad2deg(np.min(np.minimum(q - np.array(JMIN),
                                                np.array(JMAX) - q))))
    ck("够不到时不把关节顶死在限位上", not pinned, f"顶死的: {pinned or '无'}")
    ck("留出 ≥5° 余量（不是贴着驱动带子边）", margin >= 5.0, f"最紧 {margin:.2f}°")
    ck("报得出「跟不上」，上层才能重锚", s.stats["saturated"] > 8,
       f"saturated={s.stats['saturated']} (需 > slip_after=8)")

    # 反面：够得着的目标不许误报，否则会无谓地反复重锚
    T_ok = kin.fk(HOME).copy()
    T_ok[:3, 3] += [0.02, 0.01, 0.0]
    _, s2 = play(T_ok, frames=100)
    ck("够得着时不误报「跟不上」", s2.stats["saturated"] == 0,
       f"saturated={s2.stats['saturated']}")


if __name__ == "__main__":
    test_hand()
    test_imports()
    test_ik()
    test_movel_buffer()
    test_stuck()
    print("\n" + "=" * 60)
    if FAILED:
        print(f"{len(FAILED)} 项失败：" + "、".join(FAILED))
        sys.exit(1)
    print("全部通过。可以上真机了 —— 顺序见 README/回答里的分阶段清单。")
