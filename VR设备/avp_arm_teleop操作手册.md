# avp_arm_teleop 操作手册（MoveL / cart / rt 三个对照后端）

现在遥操走的是 servoJ（`avp_servoj_teleop.py`），入口见 [README](README.md)。
这份只讲 `avp_arm_teleop.py` 的三个后端，留作对照，不再往上面投入。

左手控制机械臂末端和灵巧手，键盘 ENTER 挂离合，右手**保持捏合**驱动机械臂。
松开右手或摘离合时停止臂；手指独立跟踪。没有新头显帧超过 `--max-stale` 时摘离合。

## 当前控制后端

| 后端 | 命令与逆解 | 状态 |
|---|---|---|
| `movel`（默认） | `moveAppend(list[MoveLCommand], ...)` 列表批量；控制器逆解和规划 | 已修复过滤、坐标和停机逻辑；仍为非实时排队执行 |
| `cart` | Python发布最新TCP目标，独立C++进程按1ms周期发送笛卡尔位姿；控制器逆解 | 已实现、编译和离线验证；须用匹配AR固件的SDK真机验证 |
| `rt` | 本地IK + Python关节实时流 | 保留为显式备选；其他后端失败不会自动切换到它 |

servoJ 的用法见 [遥操作_servoJ](遥操作_servoJ.md)；在它上面采数据见
[数据采集_servoJ](数据采集_servoJ.md)；三条路怎么试出来的见 [AR5遥操三条路](AR5遥操三条路.md)。

完整历史实验保存在 [调试历史](调试历史_2026-09-20.md)。旧文档中的 37ms、42.8ms 等是历史测量，
不是本次修改后的端到端延迟承诺。列表批量提交本来就受 SDK 支持，本次继续保留。

## 环境与启动

环境、头显 IP、臂和手的网络见 [README](README.md) 的「环境」「硬件依赖」两节，
命令写法和 [遥操作_servoJ](遥操作_servoJ.md) 第一节一样：在 `VR设备` 目录下，用 dexgraspvla 环境、前面加 `PYTHONPATH=`。
下面的 `192.168.x.x` 换成头显界面上显示的 IP。

MoveL：

```bash
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python avp_arm_teleop.py 192.168.x.x \
  --no-cameras --arm-backend movel --yaw -90 --hz 100 --movel-dispatch-hz 50
```

`avp_arm_teleop.py` 的 `--yaw` 默认是 0（servoJ 那边默认 -90），这里要自己给。

TCP、负载和质心填写实际测量值，例如测得 TCP 沿法兰 Z 轴 150mm 时才加 `--tcp-z 0.150`。
默认 `tcp-z=0`，受控点为法兰。`--payload 0` 表示不覆盖现有负载，不代表工具没有质量。
`--payload-com-z` 为质心位置（m）。

MoveL 沿用原来的启动回位流程；`--no-home` 跳过回位。运行中按 `h` 会先停止 MoveL，
回位请退出后运行 `home_arm.py`，避免在一个会话里同时启动两个运动路径。

C++ 笛卡尔实时后端的编译、SDK匹配和验证方法见 [cartesian_rt/README.md](cartesian_rt/README.md)。
编译匹配 SDK 后使用同一个入口：

```bash
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python avp_arm_teleop.py 192.168.x.x \
  --no-cameras --arm-backend cart \
  --cart-worker "$PWD/cartesian_rt/build-ar/cartesian_worker" \
  --yaw -90 --hz 100 --cart-speed 0.10 --cart-accel 0.4 \
  --trace /tmp/ar5-cart-trace.csv
```

`--cart-worker` 不给时默认用 `cartesian_rt/build/cartesian_worker`，那个是不带 SDK 的版本，只能做模拟运行。
`cart` 从当前实测 TCP 保持，不自动回位；起点必须在工作空间盒内（盒子见下面「工作空间」）。
初次验证先用 `--no-rotation` 分离位置问题，再加入姿态。C++ 后端不会调用本地IK。

## 本次 MoveL 修复

- 每个新头显帧进入本地缓存；`flush` 做平移/姿态过滤，再一次性提交列表。
  **整批静止或小于双阈值时不发任务**，不再无条件塞入批末点。
- 平移和旋转任一超过阈值即可通过。姿态用旋转矩阵角距离，跨 ±π 不会误判大跳变。
  小于阈值的剩余误差允许保留；死区有精度和响应代价，不能称为“零滞后”。
- 每批仍按折线长度抽稀。大跨度目标按剩余位置/姿态前导预算分步下发，最终目标保留在本地，
  后续周期继续追踪。前导满时不把全部历史继续灌进控制器。
- 设置并回读 NRT `toolset`：工具偏移明确、外部参考系为基座。反馈法兰位姿经同一个工具变换
  得到 TCP，避免拿工具目标减法兰反馈。`--flange-frame` 显式使用零工具偏移。
- 读取 SDK 错误码及 `moveExecution` 执行错误。失败向上抛出并退出，不静默换成本地 IK。
- 松开右手、摘离合、输入陈旧时 `stop()` + `moveReset()`，清除本地和控制器队列。
  重新接合需等控制器 idle，并从实测位姿重新锚定，不继续旧任务。
- 输入滤波仅随新帧更新；`flush` 可独立于新帧判断重试最新待发目标。两者仍在同一 Python
  线程里，因此 MoveL 的 SDK 阻塞没有凭空消失。`cart` 后端使用独立进程解决这个隔离问题。

列表 API 减少调用次数，但不改变 MoveL 的排队性质。`lead-ms` 只参与距离预算换算，
不能观测队列真实剩余时间，往返路径、加减速、控制器缓存仍可能产生额外延迟。
如仍出现高频起停，比较 `cart` 后端，不要为了凑批数而把静止噪声重新放进队列。

## 常用参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--hz` | 50Hz | Python读取频率；不会提高头显实际产帧率 |
| `--movel-dispatch-hz` | 50Hz | 列表批量下发频率；0=每循环尝试；20Hz最多额外等约50ms |
| `--movel-deadband` | 1mm | 相对最后保留/下发位姿的平移阈值 |
| `--movel-rot-deadband` | 0.5° | 姿态阈值；纯旋转也会通过 |
| `--movel-speed` | 250mm/s | MoveL设定速度，不代表短段一定能达到 |
| `--movel-zone` | 3 | 转弯区。控制器把它当百分比档位，不是毫米半径，3 落在 10% 档（档位见 [AR5遥操三条路](AR5遥操三条路.md) 第二条路） |
| `--movel-lead-ms` | 50ms | 与速度相乘得到位置前导预算，非延迟保证 |
| `--movel-rot-lead` | 0.15rad | 姿态前导预算 |
| `--movel-buffer` | 64 | 本地缓存容量，范围1..100 |
| `--no-movel-batch-append` | 关闭 | 显式退回逐条提交，用于对照 |
| `--movel-adaptive-speed` | 关闭 | 实验性逐批调速，不能保证消除起停 |
| `--cart-speed` / `--cart-accel` | 0.10 / 0.4 | C++线速度 m/s、加速度 m/s² |
| `--cart-angular-speed` / `--cart-angular-accel` | 0.5 / 1.0 | C++角速度 rad/s、角加速度 rad/s² |
| `--max-stale` | 0.25s | 无新输入超时；cart还检查C++反馈新鲜度 |
| `--arm-smooth` / `--arm-smooth-beta` | 1.0 / 1.5 | One-Euro参数；有滤波滞后，0关闭输入滤波 |
| `--scale` / `--rot-scale` | 1 / 1 | 位移/旋转倍率 |
| `--delta-frame` | tool | tool映射随末端转动；base映射固定于基座 |
| `--tcp-z` | 0m | 受控TCP沿法兰Z的偏移；测准后填写 |
| `--gain`、`--ik`、`--lock-j5` | 见帮助 | 仅本地IK的rt后端使用 |

MoveL运行时：`[` / `]` 调平移死区，`d` 看窗口统计，`s` 看起停统计，`a` 开关实验性自适应速度。
所有模式：ENTER挂/摘离合，右手保持捏合，q退出。`--auto-arm` 仅允许配 `--mock`。

## 坐标系与工作空间

坐标系约定、`--yaw` / `--mirror` 怎么定、`probe_axes.py` 怎么用，见 [遥操作_servoJ](遥操作_servoJ.md)
第三节第 3 步和第七节，两边是同一套映射。这里只补一点：`avp_arm_teleop.py` 的姿态来源是 `left_wrist` 旋转矩阵，
不要把历史文档中掌骨架诊断方法当成当前控制输入。

工作空间盒（基座坐标，受控点），三个后端**不一样**：

| 后端 | 盒子 | 出处 |
|---|---|---|
| `movel`、`rt` | X `[0.20,0.66]`、Y `[-0.516,0.43]`、Z `[0.08,0.65]` m | `atom_robot/ar5/config.py` 的 `workspace_min/max` |
| `cart` | X `[0.22,0.58]`、Y `[-0.516,0.28]`、Z `[0.08,0.42]` m（旧值） | 写死在 `avp_cart_teleop.py` 和 `cartesian_rt/control.hpp` 里，没跟 config 同步 |

非零TCP改变受控点，需重新核对工具、手指和台面的空间余量。
该盒子不等同于运动学可达性；姿态、关节限位和奇异问题由控制器报告。

## 离线验证

在 `VR设备` 目录下：

```bash
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python test_changes.py
rm -rf cartesian_rt/build      # 旧编译目录的 CMakeCache 还记着 Project_Atom/遥操 的旧路径，不删 cmake 会报错
cmake -S cartesian_rt -B cartesian_rt/build
cmake --build cartesian_rt/build -j2
ctest --test-dir cartesian_rt/build --output-on-failure
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python test_cartesian_backends.py
```

`cartesian_rt/build-sdk/` 有同样的旧路径问题，要重编它也得先删，见 [cartesian_rt/README.md](cartesian_rt/README.md)。
不指定 SDK 的 CMake 构建只支持 mock，真实控制请求会直接报错。
`test_cartesian_backends.py` 用 Unix socket 启动真正的 C++ 子进程，但只传 `--mock`，不连接真机。
它默认找 `cartesian_rt/build/cartesian_worker`，找不到时进程测试会明确跳过；编译在别处时用环境变量 `ATOM_CART_WORKER` 指过去。

两个终端运行完整假头显链路（模拟运行，不连臂和手）：

```bash
# 终端1
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python mock_avp.py
# 终端2：任选一个后端；假头显启动3秒后右手捏合
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python avp_arm_teleop.py 127.0.0.1 --mock --mock-hand \
  --auto-arm --no-cameras --no-home --arm-backend movel --seconds 10
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python avp_arm_teleop.py 127.0.0.1 --mock --mock-hand \
  --auto-arm --no-cameras --arm-backend cart --seconds 10 --trace /tmp/cart-mock.csv
```

Mock不证明控制器支持笛卡尔RT，也不模拟真实动力学、碰撞、奇异或MoveL短段加减速。
真机验收建议记录低速直线、往返、原地旋转和启停，比较输入、目标、下发指令与实测TCP。

## 文件

| 文件 | 用途 |
|---|---|
| `avp_arm_teleop.py` | 统一入口，MoveL / cart / 本地IK备选 |
| `movel_stream.py` | NRT列表批量、双死区、前导、工具配置、停机 |
| `avp_cart_teleop.py` / `cartesian_bridge.py` | cart前端、进程协议和记录 |
| `cartesian_rt/` | C++最新目标跟踪、SDK适配、构建说明 |
| `mock_movel.py` / `mock_avp.py` | 假控制器、假头显 |
| `test_changes.py` / `test_cartesian_backends.py` | 旧回归与新控制后端回归 |
| `home_arm.py` | 独立关节回位；不要与遥操同时运行 |
| `hand_retarget.py` / `inspire_hand6.py` | 灵巧手映射和驱动 |
| `probe_axes.py` / `test_fingers.py` | 坐标诊断 / 单独测手指 |
| `try_cartesian_rt.py` / `diag_cartesian_rt.py` | 旧Python笛卡尔RT诊断；其失败结论限定于当时SDK配置 |

手标定：`PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python avp_arm_teleop.py 192.168.x.x --calib`
（`192.168.x.x` 换成头显界面上显示的 IP）。手的温度会变化但电机不响应时，
使用 `test_fingers.py` 独立定位（用法见 [遥操作_servoJ](遥操作_servoJ.md) 第七节），不要据此调整机械臂控制参数。
