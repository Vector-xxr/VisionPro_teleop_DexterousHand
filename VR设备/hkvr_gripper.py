#!/usr/bin/env python3
"""兼容旧入口；TG9801 协议实现在 SO101 的 atom_robot 驱动里维护。直接运行只做离线自检。"""
import importlib.util as _importlib_util
from pathlib import Path as _Path

from ws_paths import AR5_PKG as _AR5_PKG  # 10-15 起和 SO101 共用 atom_robot 这一份驱动
_source = _AR5_PKG / "drivers" / "hkvr_protocol.py"
_spec = _importlib_util.spec_from_file_location("_atom_hkvr_protocol", _source)
_module = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(_selftest())
