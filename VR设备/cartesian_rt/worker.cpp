#include "control.hpp"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <sys/socket.h>
#include <unistd.h>
#ifdef ATOM_WITH_SDK
#include "rokae/robot.h"
#endif
using namespace atom;
namespace {
std::atomic_bool interrupted{false};
void signal_stop(int) { interrupted.store(true); }
struct Options {
  int fd=-1;
  bool mock=false, probe=false, enable=false, self_test=false;
  std::string robot_ip="192.168.2.160", local_ip="192.168.2.222";
  double timeout=.25, tcp_z=0, payload=0, com_z=0;
  Tracker tracker;
};
Options parse(int argc,char** argv) {
  Options o;
  for(int i=1;i<argc;++i) {
    std::string a=argv[i];
    auto val=[&](){if(++i>=argc) throw std::runtime_error("missing argument: "+a);return std::string(argv[i]);};
    if(a=="--mock") o.mock=true;
    else if(a=="--probe") o.probe=true;
    else if(a=="--enable") o.enable=true;
    else if(a=="--self-test") o.self_test=true;
    else if(a=="--fd") o.fd=std::stoi(val());
    else if(a=="--robot-ip") o.robot_ip=val();
    else if(a=="--local-ip") o.local_ip=val();
    else if(a=="--timeout") o.timeout=std::stod(val());
    else if(a=="--tcp-z") o.tcp_z=std::stod(val());
    else if(a=="--payload") o.payload=std::stod(val());
    else if(a=="--com-z") o.com_z=std::stod(val());
    else if(a=="--speed") o.tracker.vmax=std::stod(val());
    else if(a=="--accel") o.tracker.amax=std::stod(val());
    else if(a=="--angular-speed") o.tracker.wmax=std::stod(val());
    else if(a=="--angular-accel") o.tracker.alphamax=std::stod(val());
    else throw std::runtime_error("unknown argument: "+a);
  }
  for(double x:{o.timeout,o.tracker.vmax,o.tracker.amax,o.tracker.wmax,o.tracker.alphamax})
    if(!std::isfinite(x)||x<=0) throw std::runtime_error("limits/timeout must be finite and positive");
  for(double x:{o.tcp_z,o.payload,o.com_z}) if(!std::isfinite(x)) throw std::runtime_error("nonfinite configuration");
  if(o.payload<0) throw std::runtime_error("negative payload");
  return o;
}
struct Bridge {
  int fd;
  std::mutex input_mutex, output_mutex;
  Target incoming{}, current{};
  State outgoing{};
  std::atomic_bool done{false}, shutdown{false}, failed{false};
  std::thread io;
  bool needs_release=true;
  uint64_t ticks=0, late=0;
  double previous_tick=0;
  explicit Bridge(int socket):fd(socket) {
    io=std::thread([this] {
      while(!done.load()) {
        Target candidate{};
        for(int n=0;n<64;++n) {
          auto count=recv(fd,&candidate,sizeof(candidate),MSG_DONTWAIT|MSG_TRUNC);
          if(count<0) { if(errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR) shutdown=true; break; }
          if(count==0) { shutdown=true; break; }
          double t=now();
          if(count!=sizeof(candidate)||candidate.tag!=magic||!std::isfinite(candidate.stamp)||
             candidate.stamp>t+.05||!valid(matrix(candidate.pose))) continue;
          std::lock_guard<std::mutex> lock(input_mutex);
          if(candidate.seq<=incoming.seq) continue;
          incoming=candidate;
          if(candidate.flags&quit) shutdown=true;
        }
        State feedback;
        {std::lock_guard<std::mutex> lock(output_mutex); feedback=outgoing;}
        if(feedback.flags&ready) send(fd,&feedback,sizeof(feedback),MSG_DONTWAIT|MSG_NOSIGNAL);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    });
  }
  ~Bridge(){done=true;if(io.joinable())io.join();}
  bool tick(Tracker& tracker,const Matrix& actual,const std::array<double,7>& q,double timeout) {
    double t=now();
    double elapsed=previous_tick ? t-previous_tick : .001;
    if(elapsed>.0015) ++late;
    previous_tick=t;
    if(input_mutex.try_lock()) {current=incoming;input_mutex.unlock();}
    bool fresh=current.seq && t-current.stamp<=timeout && t-current.stamp>=-.05;
    if(!fresh || elapsed>.02) needs_release=true;
    if(fresh && !(current.flags&enabled)) needs_release=false;
    bool active=fresh && (current.flags&enabled) && !needs_release && !shutdown && !interrupted;
    tracker.step(matrix(current.pose),active);
    State state;
    state.seq=current.seq;state.stamp=t;
    state.flags=ready|(tracker.rest()?stopped:0)|(needs_release?latched:0)|(failed?fault:0);
    state.actual=array(actual);state.command=array(tracker.command);state.joints=q;
    state.ticks=++ticks;state.late=late;
    if(output_mutex.try_lock()){outgoing=state;output_mutex.unlock();}
    return (shutdown||interrupted) && tracker.rest();
  }
};
int self_test() {
  auto require=[](bool b,const char* m){if(!b)throw std::runtime_error(m);};
  Tracker tr;
  tr.command.block<3,1>(0,3)=Eigen::Vector3d(.4,0,.25);
  Matrix target=tr.command;target(0,3)+=.08;
  target.block<3,3>(0,0)=Eigen::AngleAxisd(.3,Eigen::Vector3d::UnitZ()).toRotationMatrix();
  for(int i=0;i<8000;++i) {
    auto v=tr.velocity,w=tr.omega;
    tr.step(target,true);
    require(tr.velocity.norm()<=tr.vmax+1e-10,"speed limit");
    require((tr.velocity-v).norm()<=tr.amax*.001+1e-10,"acceleration limit");
    require((tr.omega-w).norm()<=tr.alphamax*.001+1e-10,"angular acceleration limit");
    require(tr.omega.norm()<=tr.wmax+1e-10,"angular speed limit");
    require(valid(tr.command),"rotation orthogonality");
  }
  require((tr.command-target).norm()<1e-5,"endpoint convergence");
  target(0,3)=.3;
  for(int i=0;i<500;++i)tr.step(target,true);
  for(int i=0;i<2000;++i)tr.step(target,false);
  require(tr.rest(),"release must brake");
  std::cout<<"Cartesian tracker: limits, SO(3), convergence and brake passed\n";
  return 0;
}
#ifdef ATOM_WITH_SDK
void check(const std::error_code& ec,const char* action) {
  if(ec)throw std::runtime_error(std::string(action)+": "+ec.message());
}
int real_robot(Options& o) {
  using Robot=rokae::ATOM_ROBOT_TYPE;
  Robot robot(o.robot_ip,o.local_ip);
  std::error_code ec;
  auto info=robot.robotInfo(ec);check(ec,"robotInfo");
  std::cerr<<"Robot "<<info.type<<" axes="<<info.joint_num<<"\n";
  if(info.joint_num!=7)throw std::runtime_error("expected AR seven-axis robot");
  if(o.probe) {std::cout<<"Probe complete (no mode/power/motion commands)\n";return 0;}
  if(!o.enable)throw std::runtime_error("real motion requires --enable");
  auto rt=robot.getRtMotionController().lock();
  if(!rt)throw std::runtime_error("RT controller unavailable");
  bool streaming=false;
  try {
    robot.setRtNetworkTolerance(20,ec);check(ec,"network tolerance");
    robot.setOperateMode(rokae::OperateMode::automatic,ec);check(ec,"automatic");
    robot.setMotionControlMode(rokae::MotionControlMode::RtCommand,ec);check(ec,"RT mode");
    Matrix tool=Matrix::Identity();tool(2,3)=o.tcp_z;
    rt->setEndEffectorFrame(array(tool),ec);check(ec,"TCP");
    if(o.payload>0) {
      rokae::Load load{};load.mass=o.payload;load.cog={0,0,o.com_z};load.inertia={0,0,0};
      rt->setLoad(load,ec);check(ec,"load");
    }
    robot.startReceiveRobotState(std::chrono::milliseconds(1),{"pos_m","q_m"});
    streaming=true;
    // Read initial TCP from the same state field used for all later feedback.
    std::array<double,16> measured{};std::array<double,7> joints{};
    if(!robot.updateRobotState(std::chrono::milliseconds(100)) ||
       robot.getStateData("pos_m",measured)!=0 || robot.getStateData("q_m",joints)!=0)
      throw std::runtime_error("initial RT feedback unavailable");
    o.tracker.command=matrix(measured);
    if(!valid(o.tracker.command)||!o.tracker.in_bounds(o.tracker.command))
      throw std::runtime_error("initial TCP invalid/outside workspace; position robot before starting");
    Bridge bridge(o.fd);
    std::atomic_bool finished{false};
    // Declared after every callback capture: it stops the SDK thread before
    // finished/bridge/feedback are destroyed, including exception unwinding.
    struct LoopGuard {
      std::function<void()> stop;
      ~LoopGuard(){try{stop();}catch(...){}}
    } guard{[&]{rt->stopLoop();rt->stopMove();}};
    std::function<rokae::CartesianPosition()> callback=[&] {
      rokae::CartesianPosition out;
      try {
        if(robot.getStateData("pos_m",measured)!=0 || robot.getStateData("q_m",joints)!=0)
          throw std::runtime_error("missing feedback");
        if(bridge.tick(o.tracker,matrix(measured),joints,o.timeout)) {out.setFinished();finished=true;}
        out.pos=array(o.tracker.command);
      } catch(...) {bridge.failed=true;out.pos=array(o.tracker.command);out.setFinished();finished=true;}
      return out;
    };
    robot.setPowerState(true,ec);check(ec,"power on");
    rt->setControlLoop(callback,0,true);
    rt->startMove(rokae::RtControllerMode::cartesianPosition);
    rt->startLoop(false);
    while(!finished) {
      if(rt->hasMotionError())throw std::runtime_error("controller RT motion error");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    rt->stopLoop();rt->stopMove();
    if(bridge.failed)throw std::runtime_error("RT callback failed; inspect controller log");
    robot.stopReceiveRobotState();streaming=false;
    robot.setMotionControlMode(rokae::MotionControlMode::NrtCommand,ec);check(ec,"exit RT");
    return 0;
  } catch(...) {
    try{rt->stopLoop();rt->stopMove();}catch(...){}
    if(streaming)try{robot.stopReceiveRobotState();}catch(...){}
    throw;
  }
}
#endif
}
int main(int argc,char** argv) {
  try {
    auto o=parse(argc,argv);
    if(o.self_test)return self_test();
    if(o.fd<0 && !o.probe)throw std::runtime_error("--fd required (launch via Python bridge)");
    std::signal(SIGINT,signal_stop);std::signal(SIGTERM,signal_stop);
    if(o.mock) {
      o.tracker.command.block<3,1>(0,3)=Eigen::Vector3d(.4,0,.25);
      Bridge bridge(o.fd);
      auto deadline=std::chrono::steady_clock::now();
      while(!bridge.tick(o.tracker,o.tracker.command,{},o.timeout)) {
        deadline+=std::chrono::milliseconds(1);
        std::this_thread::sleep_until(deadline);
        if(std::chrono::steady_clock::now()>deadline+std::chrono::milliseconds(2))deadline=std::chrono::steady_clock::now();
      }
      return 0;
    }
#ifdef ATOM_WITH_SDK
    return real_robot(o);
#else
    throw std::runtime_error("mock-only build: rebuild with matching AR SDK for real robot");
#endif
  } catch(const std::exception& e) {std::cerr<<"cartesian_worker: "<<e.what()<<"\n";return 2;}
}
