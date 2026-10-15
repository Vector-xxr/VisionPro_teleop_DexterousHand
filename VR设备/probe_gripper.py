#!/usr/bin/env python3
"""认夹爪：它是什么协议、什么波特率、什么从站号。**全程只读，不发任何动作指令。**

    python probe_gripper.py                      # 默认 /dev/ttyUSB0，扫一遍
    python probe_gripper.py --port /dev/ttyUSB0 --listen-only   # 只被动听，一个字节都不发

两步：

  1. **被动监听** —— 有些夹爪上电就周期性吐状态帧。真听到东西就先看它长什么样，
     这一步连一个字节都不发。
  2. **Modbus RTU 只读查询** —— 发功能码 0x03（读保持寄存器）。
     Modbus 协议里 0x03 是只读的，不会让夹爪动。回包要 CRC 校验通过才算数，
     所以不会把噪声认成设备。

**不做的事**：不发功能码 0x06/0x10（写寄存器）、不发初始化/标定、不发任何位置指令。
认出来型号之后再写驱动，那时候才碰写操作。

常见的两指夹爪基本都是 Modbus RTU 115200 8N1，从站号 1；大寰 (DH) PGE/PGI/AG 系列、
Robotiq 2F（从站 9）、因时 EG2 都是这个路子。

本机设备为航凯 HKVR-TG9801，用户当前接武汉若比特 USB-CAN。
不能根据本脚本的 Modbus 无回复就断定硬件不支持 CAN。CAN 主动诊断请使用
VEL 的 `python -m experiments.robot.ar5.probe_hkvr_can`，可读适配器配置并查询全部接收 ID。
手册中“通过适配盒”的描述尚不能单独确定这只实物的接口版本。
"""
from __future__ import annotations

import argparse
import sys
import time

BAUDS = (115200, 57600, 38400, 19200, 9600)
SLAVES = (1, 2, 9, 0x0A, 16)
# 各家夹爪常用的只读寄存器起点。读不到就换下一个，读到什么都不解释成指令。
PROBE_REGS = (0x0000, 0x0100, 0x0200, 0x07D0, 0x03E8)


def crc16(data: bytes) -> bytes:
    """Modbus RTU 的 CRC-16，低字节在前。"""
    crc = 0xFFFF
    for ch in data:
        crc ^= ch
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return bytes((crc & 0xFF, crc >> 8))


def read_request(slave: int, reg: int, count: int = 1) -> bytes:
    """功能码 0x03，读保持寄存器。**只读**。"""
    body = bytes((slave, 0x03)) + reg.to_bytes(2, "big") + count.to_bytes(2, "big")
    return body + crc16(body)


def valid_reply(data: bytes, slave: int):
    """回包必须 CRC 通过才算数，否则就是噪声或者别的协议。"""
    if len(data) < 5 or data[0] != slave:
        return None
    if data[1] == 0x03:
        n = data[2]
        frame = data[: 3 + n + 2]
        if len(frame) == 3 + n + 2 and crc16(frame[:-2]) == frame[-2:]:
            regs = [int.from_bytes(frame[3 + 2 * i: 5 + 2 * i], "big") for i in range(n // 2)]
            return ("ok", regs, frame)
    if data[1] == 0x83:                      # 异常响应：**设备在，只是这个地址不对**
        frame = data[:5]
        if crc16(frame[:-2]) == frame[-2:]:
            return ("exception", data[2], frame)
    return None


def listen(ser, seconds: float):
    """被动听。一个字节都不发。"""
    print(f"\n  被动监听 {seconds:.0f}s（不发任何东西）...")
    end, chunks = time.time() + seconds, []
    while time.time() < end:
        n = ser.in_waiting
        if n:
            chunks.append(ser.read(n))
        time.sleep(0.02)
    data = b"".join(chunks)
    if data:
        print(f"    收到 {len(data)} 字节：{data[:64].hex(' ')}"
              + (" ..." if len(data) > 64 else ""))
        print(f"    夹爪上电就在吐状态帧。按这个帧头去对型号手册。")
    else:
        print(f"    一个字节都没有。多数夹爪是问一句答一句，这很正常。")
    return data


def scan(port: str, bauds, slaves, timeout: float):
    import serial

    found = []
    for baud in bauds:
        try:
            ser = serial.Serial(port, baud, timeout=timeout)
        except Exception as exc:                        # noqa: BLE001
            print(f"  {baud:>7} 打不开：{exc}")
            continue
        with ser:
            time.sleep(0.05)
            ser.reset_input_buffer()
            hits = []
            for slave in slaves:
                for reg in PROBE_REGS:
                    ser.reset_input_buffer()
                    ser.write(read_request(slave, reg, 1))
                    ser.flush()
                    time.sleep(0.06)
                    data = ser.read(64)
                    if not data:
                        continue
                    got = valid_reply(data, slave)
                    if got is None:
                        continue
                    if got[0] == "ok":
                        hits.append((slave, reg, got[1], got[2]))
                        print(f"  ✓ {baud} 从站 {slave} 寄存器 0x{reg:04X} = {got[1]}"
                              f"   原始 {got[2].hex(' ')}")
                    else:
                        hits.append((slave, reg, f"异常码 {got[1]}", got[2]))
                        print(f"  ~ {baud} 从站 {slave} 寄存器 0x{reg:04X} 异常码 {got[1]}"
                              f" —— **设备在**，只是这个地址不对")
            if hits:
                found.append((baud, hits))
            else:
                print(f"  {baud:>7} 无有效回包")
    return found


def can_sniff(port: str, seconds: float, baud: int = 115200) -> int:
    """被动监听 USB-CAN 适配器上的总线。**一个字节都不发。**

    夹爪（HKVR-TG9801）和原来那只因时手挂在同一个 USB-CAN 适配器上。
    适配器那层的串口帧格式是实测验证过的（见 `inspire_hand6.py` 文件头）：

        AA AA | CAN_ID 4B 小端 | 数据 8B | DLC | 00 01 00 | CKS | 55 55
        CKS = sum(frame[2:18]) & 0xFF，**在转义之前**算
        帧体（下标 2..18）里的 0x55 / 0xAA / 0xA5 前面要插一个 0xA5

    适配器是通用的，CAN_ID 和数据段怎么解释是各设备自己的事 —— 因时手那套
    「读写位 + 寄存器地址 + 设备号」的 ID 编码是**因时专有的**，别套到夹爪上。

    所以这里只做一件事：把总线上跑的帧原样解出来，看夹爪自己广播什么 ID、
    什么数据。很多 CAN 设备上电就周期性播状态，那是不用手册也能拿到的线索。
    """
    import serial

    HEAD, TAIL, ESC = b"\xAA\xAA", b"\x55\x55", 0xA5
    print(f"\n  被动监听 CAN 总线 {seconds:.0f}s（不发任何东西）...")
    try:
        ser = serial.Serial(port, baud, bytesize=8, parity="N", stopbits=1, timeout=0.1)
    except Exception as exc:                            # noqa: BLE001
        print(f"  打不开 {port}：{exc}")
        return 1

    buf, frames, raw_total = bytearray(), [], 0
    with ser:
        end = time.time() + seconds
        while time.time() < end:
            chunk = ser.read(256)
            if chunk:
                raw_total += len(chunk)
                buf += chunk
            # 按帧头帧尾切，帧长不定（转义之后 21~38），不能按固定长度对齐
            while True:
                i = buf.find(HEAD)
                if i < 0:
                    del buf[:-1]
                    break
                j = buf.find(TAIL, i + 2)
                if j < 0:
                    if i:
                        del buf[:i]
                    break
                frames.append(bytes(buf[i:j + 2]))
                del buf[:j + 2]
            time.sleep(0.01)

    print(f"    收到 {raw_total} 字节，切出 {len(frames)} 帧")
    if not frames:
        print("    总线上什么都没有。夹爪多半是问一句答一句，或者**没上 24V**")
        print("    （适配器的 USB 只供通信，夹爪的动力电源要单独接）")
        return 2

    seen = {}
    for fr in frames:
        body = bytearray()
        k, esc = 2, False
        while k < len(fr) - 2:                          # 去转义
            b = fr[k]
            if not esc and b == ESC:
                esc = True
            else:
                body.append(b)
                esc = False
            k += 1
        if len(body) < 17:
            continue
        can_id = int.from_bytes(body[0:4], "little")
        data, dlc = bytes(body[4:12]), body[12]
        cks = sum(body[0:16]) & 0xFF
        ok = cks == body[16]
        key = (can_id, dlc)
        seen.setdefault(key, []).append((data, ok))

    print(f"\n    总线上出现过的 CAN ID：")
    for (can_id, dlc), items in sorted(seen.items()):
        good = sum(1 for _, o in items if o)
        sample = items[0][0][:dlc].hex(" ")
        print(f"      0x{can_id:08X}  DLC={dlc}  {len(items)} 帧（校验过 {good}）  例 {sample}")
    print(f"\n    拿这些 ID 去对 HKVR-TG9801 的手册。**别套因时那套 ID 编码。**")
    return 0


def probe_xpanel(vout: int, slaves, regs, verbose=False) -> int:
    """走**臂的 xPanel 末端面板**探测，不是本机 USB 串口。

    法兰上装的夹爪一般不接本机 USB —— 它从臂的末端接口取 24V 和 RS485，
    信号走臂内部线缆回控制器，再由 SDK 的 `XPRWModbusRTUReg` 代发 Modbus。
    所以本机 `lsusb` 里根本不会多出设备，插上夹爪前后一模一样。

    这里只发功能码 **0x03（读保持寄存器）**，不写。但**会先给末端上 24V** ——
    夹爪上电后有的型号会自检回零，那是它自己的行为，属于正常上电动作。
    """
    import sys as _s
    from ws_paths import VEL_REPO
    _s.path.insert(0, str(VEL_REPO))
    from experiments.robot.ar5.ar5_env import AR5Env
    from experiments.robot.ar5.config import AR5Config

    cfg = AR5Config(mock=False, real_hand_in_mock=False)
    cfg.use_hand = False
    cfg.arm.use_realtime = False          # 只连，不进实时，不动臂
    cfg.cameras.sources = {}
    env = AR5Env(cfg)
    print("  连接 AR5（只连，不动臂）...", flush=True)
    env.connect()
    robot, sdkmod = env.arm._robot, env.arm.sdk
    found = []
    try:
        ec = {}
        # opt: 0 关 / 1 保持 / 2 12V / 3 24V
        robot.setxPanelRS485(vout, True, ec)
        print(f"  xPanel: 供电模式 {vout}（3=24V）+ RS485 打开  ec={ec.get('ec', 0)}")
        if ec.get("ec", 0):
            print("  ⚠ 设置没成功。这台机型可能不支持 xPanel，或者要先在示教器里开。")
        time.sleep(0.5)                   # 给夹爪上电自检的时间

        for slave in slaves:
            for reg in regs:
                arr = sdkmod.PyTypeVectorInt()
                ec2 = {}
                try:
                    robot.XPRWModbusRTUReg(slave, 0x03, reg, "int32", 1, arr, False, ec2)
                except Exception as exc:              # noqa: BLE001
                    if verbose:
                        print(f"    从站 {slave} 0x{reg:04X}: {type(exc).__name__}: {exc}")
                    continue
                code = ec2.get("ec", 0)
                vals = list(arr)
                if code == 0 and vals:
                    found.append((slave, reg, vals))
                    print(f"  ✓ 从站 {slave} 寄存器 0x{reg:04X} = {vals}")
                elif verbose:
                    print(f"    从站 {slave} 0x{reg:04X}: ec={code} vals={vals}")
    finally:
        env.close()

    print()
    if found:
        print(f"  ── 找到了 ── 从站号 {sorted({f[0] for f in found})}")
        print(f"  下一步：按型号手册对寄存器表再写驱动。**写操作还没做过。**")
        return 0
    print("  xPanel 上没读到 Modbus 设备。可能是：")
    print("    · 夹爪不是 Modbus RTU（也可能是数字 IO 式的，用 setDO 开合）")
    print("    · 从站号不在试过的范围")
    print("    · xPanel 的**波特率**不对 —— SDK 的 setxPanelRS485 不带波特率参数，")
    print("      那是在示教器里配的，要去示教器上确认它和夹爪一致")
    print("    · 这台机型不支持 xPanel")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can-sniff", action="store_true",
                   help="被动监听 USB-CAN 适配器上的总线，一个字节都不发")
    ap.add_argument("--xpanel", action="store_true",
                   help="走臂的 xPanel 末端面板探测（法兰上装的夹爪走这条）")
    ap.add_argument("--vout", type=int, default=3, choices=(0, 1, 2, 3),
                   help="xPanel 供电：0 关 / 1 保持 / 2 12V / 3 24V")
    ap.add_argument("--verbose", action="store_true", help="每个从站每个寄存器都打印")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--listen", type=float, default=2.0, help="被动监听几秒")
    ap.add_argument("--listen-only", action="store_true", help="只听，一个字节都不发")
    ap.add_argument("--baud", type=int, default=0, help="只试这一个波特率")
    ap.add_argument("--timeout", type=float, default=0.15)
    a = ap.parse_args()

    if a.can_sniff:
        return can_sniff(a.port, a.listen or 5.0, a.baud or 115200)
    if a.xpanel:
        return probe_xpanel(a.vout, SLAVES, PROBE_REGS, a.verbose)

    try:
        import serial                                    # noqa: F401
    except ImportError:
        print("  没装 pyserial。用 vel-ar5 环境跑。")
        return 1
    import serial

    print(f"  口 {a.port}")
    try:
        with serial.Serial(a.port, a.baud or 115200, timeout=a.timeout) as ser:
            listen(ser, a.listen)
    except Exception as exc:                             # noqa: BLE001
        print(f"  打不开 {a.port}：{exc}")
        print(f"  查占用：fuser -v {a.port}")
        return 1

    if a.listen_only:
        return 0

    print(f"\n  Modbus RTU 只读扫描（功能码 0x03，不会让夹爪动）...")
    found = scan(a.port, (a.baud,) if a.baud else BAUDS, SLAVES, a.timeout)

    print()
    if not found:
        print("  没扫到 Modbus 设备。可能是：")
        print("    · 不是 Modbus 协议（有些夹爪是私有 ASCII 或 CAN 透传）")
        print("    · 波特率不在 " + "/".join(str(b) for b in BAUDS))
        print("    · 从站号不在 " + "/".join(str(s) for s in SLAVES))
        print("    · 夹爪没上 24V —— **USB 只供通信，动力电源要单独接**")
        return 2
    baud, hits = found[0]
    print(f"  ── 找到了 ──")
    print(f"  波特率 {baud}   从站号 {sorted({h[0] for h in hits})}")
    print(f"  下一步：按型号手册对寄存器表，再写驱动。**写操作现在还没做过。**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
