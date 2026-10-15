#pragma once
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <time.h>

namespace atom {
using Matrix = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
constexpr uint64_t magic = 0x41544f4d43525431ULL;
constexpr uint64_t enabled = 1, quit = 2;
constexpr uint64_t ready = 1, stopped = 2, latched = 4, fault = 8;
struct Target {
  uint64_t tag=magic, seq=0, flags=0;
  double stamp=0;
  std::array<double,16> pose{};
};
struct State {
  uint64_t tag=magic, seq=0, flags=0;
  double stamp=0;
  std::array<double,16> actual{}, command{};
  std::array<double,7> joints{};
  uint64_t ticks=0, late=0;
};
static_assert(sizeof(Target)==160 && sizeof(State)==360, "IPC ABI changed");
inline double now() {
  timespec ts{}; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec*1e-9;
}
inline Matrix matrix(const std::array<double,16>& p) { return Eigen::Map<const Matrix>(p.data()); }
inline std::array<double,16> array(const Matrix& m) {
  std::array<double,16> a; std::copy(m.data(),m.data()+16,a.begin()); return a;
}
inline bool valid(const Matrix& m) {
  auto R=m.block<3,3>(0,0);
  return m.allFinite() && (m.row(3)-Eigen::RowVector4d(0,0,0,1)).norm()<1e-6 &&
    (R.transpose()*R-Eigen::Matrix3d::Identity()).norm()<1e-5 && std::abs(R.determinant()-1)<1e-5;
}
inline Eigen::Vector3d limit(Eigen::Vector3d v, double maximum) {
  double n=v.norm(); return n>maximum ? (v*(maximum/n)).eval() : v;
}
struct Tracker {
  Matrix command=Matrix::Identity();
  Eigen::Vector3d velocity=Eigen::Vector3d::Zero(), omega=Eigen::Vector3d::Zero();
  Eigen::Vector3d lower{0.22,-0.516,0.08}, upper{0.58,0.28,0.42};
  double vmax=.1, amax=.4, wmax=.5, alphamax=1., gain=12.;
  bool rest() const { return velocity.norm()<1e-6 && omega.norm()<1e-6; }
  bool in_bounds(const Matrix& m) const {
    auto p=m.block<3,1>(0,3); return (p.array()>=lower.array()).all() && (p.array()<=upper.array()).all();
  }
  void step(const Matrix& desired, bool active, double dt=.001) {
    const Eigen::Vector3d p=command.block<3,1>(0,3);
    const Eigen::Matrix3d R=command.block<3,3>(0,0);
    Eigen::Vector3d vgoal=Eigen::Vector3d::Zero(), wgoal=Eigen::Vector3d::Zero();
    if(active) {
      Eigen::Vector3d goal=desired.block<3,1>(0,3).cwiseMax(lower).cwiseMin(upper);
      Eigen::Vector3d e=goal-p;
      vgoal=limit(gain*e,std::min(vmax,std::sqrt(amax*e.norm())));
      Eigen::AngleAxisd aa(desired.block<3,3>(0,0)*R.transpose());
      wgoal=limit(gain*aa.angle()*aa.axis(),std::min(wmax,std::sqrt(alphamax*std::abs(aa.angle()))));
    }
    // Leave space to brake before the workspace boundary. Apply per-axis caps
    // before acceleration limiting; a previously valid state can brake in time.
    for(int i=0;i<3;++i) {
      double hi=std::sqrt(amax*std::max(0.,upper[i]-p[i]));
      double lo=std::sqrt(amax*std::max(0.,p[i]-lower[i]));
      vgoal[i]=std::clamp(vgoal[i],-lo,hi);
    }
    velocity+=limit(vgoal-velocity,amax*dt);
    omega+=limit(wgoal-omega,alphamax*dt);
    command.block<3,1>(0,3)=p+velocity*dt;
    double angle=omega.norm()*dt;
    if(angle>1e-14) command.block<3,3>(0,0)=Eigen::AngleAxisd(angle,omega.normalized()).toRotationMatrix()*R;
    if(!in_bounds(command)) throw std::runtime_error("Cartesian command left workspace while braking");
  }
};
}
