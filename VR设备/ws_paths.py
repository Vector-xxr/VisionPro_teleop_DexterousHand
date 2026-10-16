"""工作区里其他项目的位置 —— 不写死本机绝对路径。

从本文件往上找「同时有 算法/ 和 数据_硬件/ 的目录」当工作区根，再拼出各项目的位置。
整个工作区搬到别的目录、换用户、拷进移动硬盘都不用改；只把 VR设备 单独拷走时，
用环境变量 YUHANG_WS 指到工作区根。
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_workspace() -> Path:
    env = os.environ.get("YUHANG_WS")
    if env:
        return Path(env).expanduser().resolve()
    for p in HERE.parents:
        if (p / "算法").is_dir() and (p / "数据_硬件").is_dir():
            return p
    # 找不到就按现在的层级（数据_硬件/灵巧手遥操/VR设备）推，用到的路径不存在时由调用方报错
    return HERE.parents[2]


WS = _find_workspace()
SDK_ROOT = WS / "数据_硬件" / "xCoreSDK-Python"                  # 珞石 SDK
SO101 = WS / "数据_硬件" / "SO101"                               # atom_robot 所在目录
# AR5 驱动：2026-10-15 起和 SO101 共用 atom_robot 这一份（原来用 vel_ar5 里 9-23 的旧副本）。
# 运动学 kinematics.py / transforms.py 两份逐字节相同，换过来不影响逆解。
AR5_PKG = SO101 / "atom_robot" / "ar5"
DEXGRASPVLA = WS / "算法" / "DexGraspVLA-"
