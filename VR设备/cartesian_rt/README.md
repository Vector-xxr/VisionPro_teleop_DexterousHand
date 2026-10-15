# C++ 笛卡尔实时后端

入口：`avp_arm_teleop.py --arm-backend cart`。
Python只读取头显、映射TCP目标和驱动灵巧手；C++子进程独占机械臂连接，
使用 `cartesianPosition`，以1ms周期发送 `CartesianPosition.pos`。
没有调用本地IK，也没有使用依赖主机逆解的 FollowPosition。

## 当前验证状态

- 已通过C++跟踪器自检：速度/加速度约束、旋转正交性、收敛和减速停止。
- 已通过Python/C++真实子进程IPC测试：新目标替换、释放、超时锁定、旧时间戳、异常包。
- 已使用本机C++ SDK 0.6.0和 `xMateErProRobot` 编译并链接；这仅证明该版本API可编译。
- **未连接真机，未验证这台AR5的笛卡尔RT支持和端到端延迟。** 完整假头显联调未完成。
- 本机0.6.0 SDK没有 `ArRobot`。官方AR文档使用匹配固件的AR专用SDK及 `ArRobot`，
  不要把旧SDK编译成功当作机型兼容性结论。

## 编译

依赖：C++17、CMake、Eigen3、pthread；真实后端额外需要厂商匹配版本的C++ SDK。
下面的命令都在 `VR设备` 目录下执行。

**现有的 `build/`、`build-sdk/` 是在旧路径 `Project_Atom/遥操/VR设备` 下配置的**，CMakeCache 还记着旧路径，
直接在上面跑 cmake 会报错。要重编哪个，先把哪个删掉（编译产物不进 git，删了重编即可）。

不连接机器人、只用于离线测试的版本：

```bash
rm -rf cartesian_rt/build
cmake -S cartesian_rt -B cartesian_rt/build
cmake --build cartesian_rt/build -j2
ctest --test-dir cartesian_rt/build --output-on-failure
```

它会拒绝所有非 `--mock` 运行。`avp_arm_teleop.py --cart-worker` 不给时默认用的就是这个
`cartesian_rt/build/cartesian_worker`，所以不显式指定的话只能做模拟运行。

匹配AR SDK的正式构建（从 `VR设备` 目录执行）：

```bash
cmake -S cartesian_rt -B cartesian_rt/build-ar \
  -DXCORE_SDK_ROOT=/路径/匹配固件的AR-SDK \
  -DXCORE_SDK_LIBRARY=/路径/匹配固件的AR-SDK/lib/Linux/x86_64/libxCoreSDK.so \
  -DXCORE_ROBOT_TYPE=ArRobot
cmake --build cartesian_rt/build-ar -j2
```

不要混用不同版本的头文件和动态库。某些SDK动态库依赖同目录 `libxMateModel.a`，
CMake会自动链接；也可通过 `XCORE_MODEL_LIBRARY` 显式指定。这是SDK链接依赖，
本程序不调用该库的逆解接口。

本机已有0.6.0 SDK的兼容性测试构建（`build-sdk/`）。**这份 C++ SDK 不在工作区里**，在本机 `~/桌面/xCoreSDK_cpp-v0.6.0./`
下（目录名末尾就带一个点），不随工作区拷走；工作区里的 `数据_硬件/xCoreSDK-Python` 是 Python 绑定，不能拿来编这个。
重编时先删旧目录，SDK 位置换成实际所在：

```bash
rm -rf cartesian_rt/build-sdk
SDK=~/桌面/xCoreSDK_cpp-v0.6.0./xCoreSDK_cpp-v0.6.0
cmake -S cartesian_rt -B cartesian_rt/build-sdk \
  -DXCORE_SDK_ROOT="$SDK" \
  -DXCORE_SDK_LIBRARY="$SDK/lib/Linux/x86_64/libxCoreSDK.so.0.6.0" \
  -DXCORE_ROBOT_TYPE=xMateErProRobot
cmake --build cartesian_rt/build-sdk -j2
```

## 运行

先退出现有遥操/其他机械臂控制程序，再启动下面的其中一条命令。
可先查询机型；probe建立连接并读取robotInfo，不发送模式、上电或运动指令：

```bash
./cartesian_rt/build-sdk/cartesian_worker --probe \
  --robot-ip 192.168.2.160 --local-ip 192.168.2.222
```

真实遥操（换为你实际编译的worker路径；`192.168.x.x` 换成头显界面上显示的 IP）：

```bash
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python avp_arm_teleop.py 192.168.x.x --no-cameras --arm-backend cart \
  --cart-worker "$PWD/cartesian_rt/build-sdk/cartesian_worker" \
  --robot-ip 192.168.2.160 --local-ip 192.168.2.222 \
  --yaw -90 --hz 100 --no-rotation \
  --cart-speed 0.10 --cart-accel 0.4 --trace /tmp/ar5-cart.csv
```

ENTER挂离合，右手保持捏合，左手控制位移。先验证位置，再去掉 `--no-rotation` 验证姿态。
TCP、质量和质心通过 `--tcp-z`、`--payload`、`--payload-com-z` 填写测量值。
起点必须在基座坐标工作空间盒 `[0.22,-0.516,0.08]` 到 `[0.58,0.28,0.42]` m 内。
这个盒子写死在 `control.hpp`（`lower` / `upper`）和 `../avp_cart_teleop.py` 的钳位里，是旧值；
`atom_robot/ar5/config.py` 里的 `workspace_min/max` 已经放大到 `[0.20,-0.516,0.08]` 到 `[0.66,0.43,0.65]` m，
MoveL 和 servoJ 用的是新盒子，cart 这条没跟着改。
本后端从当前位置保持，**不自动回位**；需要回位时先退出，再单独运行 `home_arm.py`。

## 控制和故障行为

- IPC为本机Unix数据报，消息携带序号、单调时钟时间戳和4×4行优先矩阵。
  接收线程更新最新目标；SDK回调用try_lock读取快照，不执行socket等待、打印或文件写入。
- 控制器目标是基座下TCP绝对位姿。位置和SO(3)旋转跟踪分别限制速度与加速度。
  该跟踪器不保证jerk连续，也不保证每段是精确的几何直线；控制器仍负责可达性和运动保护。
- Python只有新头显帧才更新使能目标的时间戳，重复轮询不会伪造新帧。
- 释放离合或超时后减速到零，不继续追逐最后的远端目标。超时锁定后需释放使能再重新接合。
- 输入或C++反馈超时、SDK错误会停止当前后端，不自动换成本地IK。
- 普通Linux上1ms是请求周期，不是硬实时保证。反馈带有tick和late计数；超过20ms的
  回调间隔会触发输入锁定，仍应核对控制器RT通信错误和实测周期分布。
- CSV记录主机时钟、最近输入接收时间、反馈时间、使能、发送/确认序号、周期统计，
  以及原始手腕、映射目标、发送指令、实测TCP的完整矩阵。
  接收时间不是头显曝光时间，不能由该CSV直接推导完整传感器端到端延迟。

若命令矩阵在变化而实测TCP不动，请保留CSV和控制器日志，核对AR固件/SDK匹配。
不要据此增加位移幅度或解除运动保护。

官方依据：[AR机型专栏](https://docs.rokae.com/docs/SDK--0-7-x-AR/cpp/实时模式常见问题/cpp_rt_ar_special/)。
