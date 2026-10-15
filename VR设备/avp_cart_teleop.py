"""Vision Pro + hand frontend for the C++ Cartesian RT backend.

Invoked by avp_arm_teleop.py --arm-backend cart. The C++ child is the only
process that opens a robot connection. No local IK or Python servo sender.
"""
import csv
import select
import sys
import termios
import time
from pathlib import Path
import numpy as np
from cartesian_bridge import CartesianBridge, STOPPED, LATCHED
from pose_math import valid_transform


def run(args):
    from avp_arm_teleop import (Calibration, Retargeter, HandTracker, OneEuro,
        filter_rotation, connect_streamer, yaw_matrix, mirror_matrix,
        matrix_to_axis_angle, axis_angle_to_matrix, PINCH_ON, PINCH_OFF)
    from inspire_hand6 import InspireHand6, MockSerial, DEFAULT_LIMITS, INDEX, THUMB_BEND, NUM_DOF
    calib = Calibration.load(Path(args.calib_file)) if Path(args.calib_file).exists() else Calibration()
    streamer = connect_streamer(args)
    if streamer is None:
        return 1
    worker = hand = tracker = log_file = None
    old_term = None
    try:
        worker = CartesianBridge(args.cart_worker, mock=args.mock,
            robot_ip=args.robot_ip, local_ip=args.local_ip, timeout=args.max_stale,
            tcp_z=0. if args.no_tcp or args.flange_frame else args.tcp_z,
            payload=0. if args.no_tcp else args.payload, com_z=args.payload_com_z,
            speed=args.cart_speed, accel=args.cart_accel,
            angular_speed=args.cart_angular_speed, angular_accel=args.cart_angular_accel)
        print('[cart] 等待C++控制进程；从当前TCP保持，不自动回位，不使用本地IK', flush=True)
        state = worker.wait_ready()
        print('[cart] TCP', np.round(state['actual'][:3,3]*1000,2), 'mm；ENTER挂离合，右手捏合驱动，q退出')
        if args.mock_hand:
            hand = InspireHand6(port='(mock)', serial_factory=MockSerial)
        elif not args.no_hand and (not args.mock or args.real_hand):
            hand = InspireHand6(port=args.hand_port)
        if hand is not None:
            hand.connect()
            for i in range(NUM_DOF):
                if i in (INDEX, THUMB_BEND):
                    calib.limits[i] = hand.limits[i] = (0,1000)
                else:
                    lo, hi = calib.limits.get(i, DEFAULT_LIMITS[i])
                    hlo,hhi = hand.limits[i]
                    hand.limits[i] = (max(lo,hlo),min(hi,hhi))
            # 假手（MockSerial）读数恒定，「读数一直不变 = 控制循环没在跑」的判据对它不成立；真手照旧不过就退出
            ok, why = (True, '') if args.mock_hand else hand.liveness()
            if not ok:
                raise RuntimeError(f'灵巧手未就绪: {why}')
            hand.set_force(args.hand_force)
            hand.set_speed(args.hand_speed)
            hand.start_writer()
        retarget = Retargeter(calib, smooth=args.smooth, pinch_gate=args.pinch_gate)
        tracker = HandTracker(streamer, retarget, hand, args.hand_hz, args.freeze_thumb_rot, args.hand_debug)
        tracker.start()
        interactive = sys.stdin.isatty()
        if interactive:
            import tty
            old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        if args.trace:
            log_file = open(args.trace, 'w', newline='')
            writer = csv.writer(log_file)
            writer.writerow(['time','input_received','state_time','enabled','seq','ack','ticks','late']+
                            [f'{name}_{i}' for name in ('raw','target','command','actual') for i in range(16)])
        A = yaw_matrix(args.yaw) @ mirror_matrix(args.mirror)
        filt = OneEuro(args.arm_smooth,args.arm_smooth_beta) if args.arm_smooth>0 else None
        rfilt = OneEuro(args.arm_smooth,args.arm_smooth_beta) if filt else None
        armed, pinch, ref, previous = bool(args.auto_arm), False, None, None
        last_frame_time, last_print, started = -float('inf'), 0., time.monotonic()
        target, raw = state['actual'].copy(), np.eye(4)
        while not args.seconds or time.monotonic()-started < args.seconds:
            now = time.monotonic()
            state = worker.poll()
            if now-state['stamp'] > args.max_stale:
                raise RuntimeError('C++反馈超时，停止遥操')
            if interactive and select.select([sys.stdin],[],[],0)[0]:
                key = sys.stdin.read(1)
                if key=='q': break
                if key in ('\r','\n'):
                    armed, ref = not armed, None
                    print(f'\n[离合] {"挂上" if armed else "摘下，减速停止"}')
                if key.lower()=='h':
                    armed, ref = False, None
                    print('\n[cart] 已摘离合；回位请退出本程序，再运行 home_arm.py')
            frame = streamer.latest
            fresh = frame is not None and frame is not previous
            if fresh:
                previous, last_frame_time = frame, now
                data = streamer.get_latest()
                if data is None:
                    continue
                raw = valid_transform(np.asarray(data['left_wrist']).reshape(4,4))
                hand_pose = raw.copy()
                if filt:
                    hand_pose[:3,3] = filt(raw[:3,3],now)
                    hand_pose[:3,:3] = filter_rotation(rfilt,raw[:3,:3],now)
                distance = float(data['right_pinch_distance'])
                pinch = distance < (PINCH_OFF if pinch else PINCH_ON)
            input_ok = now-last_frame_time <= args.max_stale
            engage = armed and pinch and input_ok
            # A timeout requires an explicit release before continuing. Do not
            # automatically unlatch on stream recovery while still pinching.
            if state['flags'] & LATCHED and engage:
                armed, engage, ref = False, False, None
                print('\n[cart] 输入/调度超时已锁定，请松开右手并重新挂离合')
            if not engage:
                ref = None
                worker.publish(target, False)
            elif fresh:
                if ref is None:
                    if not state['flags'] & STOPPED:
                        worker.publish(target, False)
                        time.sleep(1/args.hz)
                        continue
                    ref = (hand_pose.copy(),state['actual'].copy())
                h0, robot0 = ref
                follow = state['command'][:3,:3] @ robot0[:3,:3].T if args.delta_frame=='tool' else np.eye(3)
                target = robot0.copy()
                target[:3,3] += follow @ A @ (hand_pose[:3,3]-h0[:3,3])*args.scale
                if args.with_rotation:
                    rel = A @ (hand_pose[:3,:3] @ h0[:3,:3].T) @ A.T
                    if args.delta_frame=='tool':rel=follow @ rel @ follow.T
                    rot = matrix_to_axis_angle(rel)*args.rot_scale
                    if args.rot_mode=='roll':
                        axis=robot0[:3,2];rot=axis*float(rot@axis)
                    target[:3,:3] = axis_angle_to_matrix(rot) @ robot0[:3,:3]
                target[:3,3] = np.clip(target[:3,3],[.22,-.516,.08],[.58,.28,.42])
                worker.publish(target, True, stamp=last_frame_time)
            # Never refresh an enabled target's timestamp on a duplicated frame.
            if log_file:
                writer.writerow([now,last_frame_time,state['stamp'],int(engage),worker.seq,state['seq'],state['ticks'],state['late']]+
                                [float(v) for t in (raw,target,state['command'],state['actual']) for v in t.ravel()])
            if args.print_hz>0 and now-last_print>=1/args.print_hz:
                last_print=now
                error=np.linalg.norm(target[:3,3]-state['actual'][:3,3])*1000
                print(f'\r[cart] 接合={engage} TCP误差={error:.1f}mm tick={state["ticks"]} late={state["late"]}  ',end='',flush=True)
            time.sleep(max(0.,1/args.hz-(time.monotonic()-now)))
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if old_term is not None:termios.tcsetattr(sys.stdin,termios.TCSADRAIN,old_term)
        if worker is not None:worker.close()
        if tracker is not None:tracker.stop();tracker.join(timeout=1.)
        if hand is not None:hand.stop_writer();hand.close()
        if log_file:log_file.close()
        print('\n[cart] 已结束')
