"""Offline regressions: queued MoveL and actual Python/C++ IPC (no robot connection)."""
import os
from pathlib import Path
import time
import unittest
import numpy as np
from mock_movel import MockMoveLRobot, MockMoveLSdk
from movel_stream import MoveLStreamer
from pose_math import rotation_distance
from cartesian_bridge import CartesianBridge, STOPPED, LATCHED, TARGET, MAGIC


class MoveLTests(unittest.TestCase):
    def make(self, **kwargs):
        r = MockMoveLRobot(start=[.4,0,.25])
        s = MoveLStreamer(r, MockMoveLSdk, speed=250, lead_ms=50, **kwargs)
        s.start(log=lambda *_: None)
        return r,s

    def test_stationary_and_subthreshold_do_not_append(self):
        r,s=self.make(min_step_mm=3)
        p=r.pos.copy()
        for _ in range(100):s.update(p)
        for d in (.0001,-.0001,.001,-.001):s.update(p+[d,0,0])
        self.assertEqual(r.n_append,0)
        self.assertEqual(r.n_start,0)

    def test_slow_motion_accumulates_across_flushes(self):
        r,s=self.make(min_step_mm=3)
        p=r.pos.copy()
        for i in range(1,13):s.update(p+[i*.00075,0,0])
        self.assertEqual(r.n_append,3)
        np.testing.assert_allclose(s._last_target,p+[.009,0,0])

    def test_list_batch_and_rotation(self):
        r,s=self.make(min_step_mm=1)
        seen=[];orig=r.moveAppend
        def append(cmd,*args):seen.append(cmd);return orig(cmd,*args)
        r.moveAppend=append
        p=r.pos.copy()
        for i in range(1,5):s.submit(p+[i*.002,0,0],[0,0,i*.02])
        s.flush()
        self.assertEqual(len(seen),1)
        self.assertIsInstance(seen[0],list)
        self.assertEqual(len(seen[0]),4)
        r2,s2=self.make()
        s2.update(r2.pos,[0,0,.1])
        self.assertEqual(r2.n_append,1)
        s2.update(r2.pos,[0,0,.1+2*np.pi])
        self.assertEqual(r2.n_append,1)

    def test_large_target_and_rotation_are_bounded_and_retained(self):
        r,s=self.make()
        initial=r.pos.copy()
        s.update(initial+[.1,0,0],[0,0,1.])
        self.assertLessEqual(np.linalg.norm(s._last_target-initial),s.max_lead+1e-9)
        self.assertLessEqual(rotation_distance([0,0,0],s._last_rpy),s.max_rotation_lead+1e-9)
        np.testing.assert_allclose(s._buf[-1][0],initial+[.1,0,0])
        # Simulate perfect execution, then retry WITHOUT a new head-tracking frame.
        for _ in range(30):
            r.moveReset({});r.pos=s._last_target.copy();r.rpy=s._last_rpy.copy()
            s.flush()
        np.testing.assert_allclose(s._last_target,initial+[.1,0,0],atol=1e-8)
        self.assertLess(rotation_distance(s._last_rpy,[0,0,1.]),1e-7)

    def test_tcp_offset_and_toolset_readback(self):
        r,s=self.make(tcp_z=.15)
        r.rpy=[0,np.pi/2,0]
        self.assertTrue(s.configure(log=lambda *_: None))
        s.start(log=lambda *_:None)
        np.testing.assert_allclose(s.tcp(),r.pos+[.15,0,0],atol=1e-12)
        s.update(s.tcp(),r.rpy)
        self.assertEqual(r.n_append,0)
        np.testing.assert_allclose(r._toolset.end.trans,[0,0,.15])
        np.testing.assert_allclose(r._toolset.ref.trans,[0,0,0])

    def test_release_clears_both_queues(self):
        r,s=self.make()
        s.update(r.pos+[.01,0,0])
        s.submit(r.pos+[.02,0,0])
        s.suspend()
        self.assertFalse(r.running)
        self.assertEqual(len(r.queue),0)
        self.assertEqual(s.pending_points(),0)
        self.assertEqual(s.update(r.pos+[.1,0,0]),'stopped')
        s.start(log=lambda *_:None)
        s.update(s.tcp())
        self.assertEqual(s.stats['appended'],1)

    def test_sdk_state_error_not_idle(self):
        r,s=self.make()
        def error(ec):ec['ec']=-1;return MockMoveLSdk.OperationState.idle
        r.operationState=error
        with self.assertRaises(RuntimeError):s.is_idle()

    def test_execution_error_is_reported_and_clears_queue(self):
        from types import SimpleNamespace
        r,s=self.make()
        s.sdk=SimpleNamespace(Event=SimpleNamespace(moveExecution=1),
            OperationState=MockMoveLSdk.OperationState, CoordinateType=MockMoveLSdk.CoordinateType)
        r.queryEventInfo=lambda event,ec:{'error':{'ec':42},'remark':'unreachable'}
        with self.assertRaisesRegex(RuntimeError,'unreachable'):s.flush()
        self.assertFalse(s._running)
        self.assertEqual(s.stats['execution_errors'],1)

    def test_invalid_target_rejected(self):
        r,s=self.make()
        with self.assertRaises(ValueError):s.submit([np.nan,0,0])
        with self.assertRaises(ValueError):s.submit([0,0,0],[0,0,np.inf])
        self.assertEqual(r.n_append,0)


WORKER=os.environ.get('ATOM_CART_WORKER',str(Path(__file__).parent/'cartesian_rt/build/cartesian_worker'))
@unittest.skipUnless(Path(WORKER).is_file(),'build C++ worker or set ATOM_CART_WORKER')
class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.b=CartesianBridge(WORKER,mock=True,timeout=.12)
        self.addCleanup(self.b.close)
        self.state=self.b.wait_ready()
        self.pose=self.state['actual'].copy()
        self.pump(.03,False)

    def pump(self,seconds,enabled,pose=None):
        end=time.monotonic()+seconds
        pose=self.pose if pose is None else pose
        while time.monotonic()<end:
            if enabled is not None:self.b.publish(pose,enabled)
            time.sleep(.01)
            self.state=self.b.poll()
        return self.state

    def test_pose_tracking_and_release(self):
        target=self.pose.copy();target[0,3]+=.02
        state=self.pump(1.,True,target)
        self.assertLess(abs(state['actual'][0,3]-target[0,3]),.001)
        target[0,3]+=.05
        self.pump(.2,True,target)
        state=self.pump(.5,False,target)
        self.assertTrue(state['flags']&STOPPED)
        self.assertLess(state['actual'][0,3],target[0,3]-.005)

    def test_timeout_requires_release(self):
        target=self.pose.copy();target[1,3]+=.06
        self.pump(.2,True,target)
        state=self.pump(.5,None)
        self.assertTrue(state['flags']&LATCHED)
        self.assertTrue(state['flags']&STOPPED)
        held=state['actual'].copy()
        self.pump(.2,True,target)
        np.testing.assert_allclose(self.state['actual'],held,atol=1e-8)
        self.pump(.05,False)
        state=self.pump(.3,True,target)
        self.assertGreater(state['actual'][1,3],held[1,3]+.001)

    def test_stale_timestamp_not_refreshed_by_new_sequence(self):
        target=self.pose.copy();target[0,3]+=.1
        old=time.monotonic()-1
        for _ in range(30):
            self.b.publish(target,True,stamp=old)
            time.sleep(.01)
        state=self.b.poll()
        np.testing.assert_allclose(state['actual'],self.pose,atol=1e-8)
        self.assertTrue(state['flags']&LATCHED)

    def test_malformed_packet_does_not_replace_target(self):
        self.b.socket.send(TARGET.pack(MAGIC,self.b.seq+1000,1,time.monotonic(),*([float('nan')]*16)))
        self.pump(.04,False)
        np.testing.assert_allclose(self.state['actual'],self.pose,atol=1e-8)
        self.assertLess(self.state['seq'],1000)

    def test_latest_goal_replaces_old_goal(self):
        first=self.pose.copy();first[0,3]+=.08
        second=self.pose.copy();second[1,3]+=.01
        self.pump(.05,True,first)
        state=self.pump(1.,True,second)
        self.assertLess(np.linalg.norm(state['actual'][:3,3]-second[:3,3]),.001)


if __name__=='__main__':unittest.main(verbosity=2)
