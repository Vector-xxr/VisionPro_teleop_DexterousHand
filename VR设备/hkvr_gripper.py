#!/usr/bin/env python3
"""兼容旧入口；TG9801 协议实现在 SO101/VEL 驱动中维护。直接运行只做离线自检。"""
import importlib.util as _importlib_util
from pathlib import Path as _Path

_source = (_Path(__file__).resolve().parents[3] / "算法/VEL/vel_ar5/openvla-energy/experiments/robot/ar5/drivers/hkvr_protocol.py")
_spec = _importlib_util.spec_from_file_location("_atom_hkvr_protocol", _source)
_module = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(_selftest())
