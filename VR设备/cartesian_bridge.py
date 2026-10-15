"""Latest-target IPC to the dedicated C++ robot owner. No Python SDK calls."""
from __future__ import annotations
import socket
import struct
import subprocess
import time
from pathlib import Path
import numpy as np
from pose_math import valid_transform

MAGIC = 0x41544F4D43525431
TARGET = struct.Struct('=QQQd16d')
STATE = struct.Struct('=QQQd39dQQ')
READY, STOPPED, LATCHED, FAULT = 1, 2, 4, 8


class CartesianBridge:
    def __init__(self, executable, *, mock=False, robot_ip='192.168.2.160',
                 local_ip='192.168.2.222', timeout=.25, tcp_z=0., payload=0.,
                 com_z=0., speed=.1, accel=.4, angular_speed=.5, angular_accel=1.):
        executable = Path(executable).resolve()
        if not executable.is_file():
            raise FileNotFoundError(f'C++ worker missing: {executable}; see cartesian_rt/README.md')
        self.socket, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.seq, self.feedback, self.process = 0, None, None
        self.last_pose = np.eye(4)
        args = [str(executable), '--fd', str(child.fileno()), '--mock' if mock else '--enable']
        for key, value in {'robot-ip': robot_ip, 'local-ip': local_ip, 'timeout': timeout,
                           'tcp-z': tcp_z, 'payload': payload, 'com-z': com_z,
                           'speed': speed, 'accel': accel, 'angular-speed': angular_speed,
                           'angular-accel': angular_accel}.items():
            args.extend(['--'+key, str(value)])
        try:
            self.process = subprocess.Popen(args, pass_fds=(child.fileno(),))
        except BaseException:
            self.socket.close()
            raise
        finally:
            child.close()

    def publish(self, pose, enabled=False, *, stamp=None, quit=False):
        pose = valid_transform(pose)
        self.last_pose = pose
        stamp = time.monotonic() if stamp is None else float(stamp)
        if not np.isfinite(stamp):
            raise ValueError('invalid input timestamp')
        self.seq += 1
        packet = TARGET.pack(MAGIC, self.seq, int(enabled) | (2 if quit else 0), stamp, *pose.ravel())
        try:
            self.socket.send(packet)
            return True
        except BlockingIOError:
            # Never wait for a stale target to be sent. The next update replaces it.
            return False

    def poll(self):
        for _ in range(256):
            try:
                raw = self.socket.recv(STATE.size+1)
            except BlockingIOError:
                break
            if len(raw) != STATE.size:
                continue
            data = STATE.unpack(raw)
            if data[0] != MAGIC:
                continue
            self.feedback = {'seq': data[1], 'flags': data[2], 'stamp': data[3],
                             'actual': valid_transform(data[4:20]),
                             'command': valid_transform(data[20:36]),
                             'joints': np.asarray(data[36:43]),
                             'ticks': data[43], 'late': data[44]}
        if self.process.poll() is not None:
            raise RuntimeError(f'C++ controller exited with code {self.process.returncode}; see stderr')
        if self.feedback is not None and self.feedback['flags'] & FAULT:
            raise RuntimeError('C++ controller reported a motion fault')
        return self.feedback

    def wait_ready(self, timeout=20.):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            self.publish(self.last_pose, False)
            state = self.poll()
            if state is not None and state['flags'] & READY:
                self.last_pose = state['actual'].copy()
                return state
            time.sleep(.01)
        raise TimeoutError('C++ controller initialization timed out')

    def close(self):
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                try:
                    self.publish(self.last_pose, False, quit=True)
                except OSError:
                    pass
                try:
                    self.process.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    # SIGTERM requests the same controlled brake in the worker.
                    self.process.terminate()
                    self.process.wait(timeout=5.)
        finally:
            self.socket.close()
