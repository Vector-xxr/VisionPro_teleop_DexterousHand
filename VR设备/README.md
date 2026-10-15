# Vision Pro 遥操 AR5 + 灵巧手

用 Apple Vision Pro 的手部追踪遥操 AR5 七轴臂和因时 RH56 灵巧手：左手带着臂末端和五根手指走，右手捏合当离合；
臂走 servoJ，逆解在本地做；可以边遥操边录数据。

---

## 当前状态

- **主力是 servoJ**（`avp_servoj_teleop.py`）。基线是 2026-10-10 那次 275 秒的任务（`logs/servoj_20261010_140947.log`）：
  逆解失败 0/25589，端到端延迟 72ms，对齐后残差 1.9mm，手 20.4Hz、0 错。
- **2026-10-15 AR5 驱动换成 `数据_硬件/SO101/atom_robot`**，同日真机验收通过（`logs/servoj_20261015_151836.log`，34 秒）：
  启动回位正常，逆解失败 0，端到端 72ms，残差 1.5mm，手 20.1Hz、0 错；`home_arm.py` 正常。
- 手从 10-09 起默认走网口 Modbus TCP。
- **servoJ 的 `--record` 还没真正采过数据**：本机 `~/atom_episodes`、`~/vel_episodes` 里的 497 个 npz 都没有它必写的 `ee_target` 字段。

---

## 启动

```bash
cd ~/yuhang_workspace/数据_硬件/灵巧手遥操/VR设备
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python avp_servoj_teleop.py 192.168.x.x --planner spline --lag-ms 50 \
    --rot-mode beside --ws-max-z 0.60 --w-psi 0.5 --max-step-rad 0.015
```

- `192.168.x.x` 换成头显 Tracking Streamer 界面上显示的 IP，每次可能变。
- 这几个参数为什么要给、每个参数的原理，见 [遥操作_servoJ](遥操作_servoJ.md)。
- **一启动臂就会先关节回位**，3 秒内 Ctrl+C 可取消，加 `--no-home` 跳过。

按键：

| 键 | 作用 |
|---|---|
| ENTER | 挂 / 摘离合，打印 `[离合] 挂上` 才算生效 |
| 右手捏合 | 挂上离合后保持捏住，臂才跟；松开即停 |
| `h` | 停下、退出 servoJ、关节回位，然后程序退出 |
| `q` | 退出 |

---

## 回位和救援

| 情况 | 做法 |
|---|---|
| 遥操中想回起始姿态 | 按 `h` |
| 遥操没在跑 | `home_arm.py`（排队式 MoveAbsJ，不开实时模式）。**不要和遥操同时跑** |
| 运动被静默拒绝（`reachTarget=False` 但 `ec=0`）= 关节在软限位外 | 先 `softlimit_rescue.py --dry-run` 看计划，再 `softlimit_rescue.py`（要输入 yes）：临时关软限位、把超限关节转回限位内、收尾恢复软限位；之后再跑 `home_arm.py`。这种情况下直接跑 `home_arm.py` 会**误报「完成」** |
| 急停、上不了电、手不回包等 | 见 [遥操作_servoJ](遥操作_servoJ.md) 第七节 |

都在 `VR设备` 目录下跑，例如：

```bash
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python home_arm.py
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python softlimit_rescue.py --dry-run
```

---

## 采集（概要）

完整流程见 [数据采集_servoJ](数据采集_servoJ.md)。

1. 看相机：用 SO101 的 `atom_robot.ar5.view_camera`（要 `cd 数据_硬件/SO101` 再 `-m` 运行），**不要用本目录的 `view_cameras.py`**（它用 V4L2 去开所有相机，碰了 Orbbec 会和 OBSENSOR 互相污染）。
2. 主命令加 `--record --out-dir … --task "…"`，空格开录，`s` 存并标成功，`←` 存但不标成功，退格丢弃。
3. 采完：`qc_episodes.py` 自动检查 → `replay_episode.py` 回放 → `manage_episodes.py` 删和编号 →
   `to_lerobot_dex.py`（lerobot 环境）或 `to_dexgraspvla.py` 转数据集。

`to_dexgraspvla.py` 被 `算法/DexGraspVLA/make_h100_dataset.sh` 按相对路径调用，**不能挪位置、不能改名**。

---

## 模拟运行

**servoJ 没有模拟模式**：`avp_servoj_teleop.py` 没有 `--mock`，一启动就连真臂。

不连硬件只能跑 `avp_arm_teleop.py` 的假头显链路（MoveL 后端，假臂、假手），命令见
[avp_arm_teleop操作手册](avp_arm_teleop操作手册.md)「离线验证」最后一段。它验证的是头显 → 映射 → 手 → MoveL 那条链，不是 servoJ。

---

## 环境

- conda 环境 `dexgraspvla`，命令前一律加 `PYTHONPATH=`（`~/.bashrc` 加载的 ROS 会污染它）。
  遥操要的 `pymodbus`（2.5.3）、`avp_stream`、`grpc`、`ruckig` 都已装在里面。
- `to_lerobot_dex.py` 用 `lerobot` 环境。
- `requirements.txt` 已过时（只列了 vuer / numpy / cryptography），别照它装。
- AR5 驱动用工作区里的 `数据_硬件/SO101/atom_robot`，由 `ws_paths.py` 往上找工作区根自动定位。
  单独拷走这个目录、换机器的规则见 [工作区说明](../../../工作区说明.md) 第 3、6 节。

---

## 离线测试

不连任何硬件，在 `VR设备` 目录下：

```bash
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python test_changes.py   # 最后打印「全部通过」
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python bench_ik.py       # 正常结束
```

- `test_changes.py` 测的是手的握拳耦合、`AR5OptIK` 对驱动的契约、模块导入、MoveL 缓存，**测不到 servoJ 主链**
  （选解层、延迟样条、beside）。servoJ 主链没有离线测试，改了只能上真机看退出统计。
- C++ 笛卡尔后端的测试（要先编译）和假头显链路见 [avp_arm_teleop操作手册](avp_arm_teleop操作手册.md)「离线验证」。
- 正常结果和真机验收基准见 [工作区说明](../../../工作区说明.md) 第 5 节。

---

## 文件地图

| 分组 | 文件 |
|---|---|
| 主链 | `avp_servoj_teleop.py`（主程序）、`ar5_poe_ik.py` + `paden_kahan.py`（闭式逆解）、`ar5_ik_select.py` + `ar5_sew.py`（选解层）、`ar5_dh.json`、`hand_retarget.py` + `hand_calib.json`、`inspire_hand6.py`（手的驱动）、`gripper.py`、`ws_paths.py`、`vel_kin.py` |
| 主链依赖的对照方案 | `avp_arm_teleop.py`、`avp_movel_teleop.py`：servoJ 主程序从这两个文件里导入滤波器、连头显、日志等，**不能删** |
| 数采 | `episode_recorder.py`、`qc_episodes.py`、`replay_episode.py`、`manage_episodes.py`、`to_lerobot_dex.py`、`to_dexgraspvla.py` |
| 回位 / 救援 | `home_arm.py`、`softlimit_rescue.py`、`jog_joint.py`、`drag_back.py`、`goto_zero.py` |
| 标定 / 诊断 | `probe_axes.py`（定 `--yaw` / `--mirror`）、`test_fingers.py`、`probe_servoj.py`、`check_fk.py`、`read_dh.py`、`probe_hand_tcp.py`、`probe_hand_data.py` |
| 其他对照方案 | `movel_stream.py`、`mock_movel.py`、`avp_cart_teleop.py`、`cartesian_bridge.py`、`pose_math.py`、`cartesian_rt/`；`try_*`、`diag_*` 是探路脚本 |
| 测试 / 模拟 | `test_changes.py`、`test_cartesian_backends.py`、`bench_ik.py`、`mock_avp.py` |
| 夹爪 | `hkvr_gripper.py`（旧入口，直接运行只做离线自检）、`probe_gripper.py` |
| 早期方案 | `avp_teleop.py`、`vision_pro_pose.py` + `gen_cert.py`（vuer 那条路）、`view_cameras.py`、`probe_cameras.py`（都走 V4L2，别拿来看 Orbbec） |
| 日志 | `logs/servoj_<时间>.log`：每次运行自动写，不进 git |

---

## 硬件依赖

| 设备 | 地址 / 说明 |
|---|---|
| AR5 控制器 | `192.168.2.160`，本机网卡 `192.168.2.222`（`--robot-ip` / `--local-ip` 可改） |
| 灵巧手 RH56 | 默认网口 `tcp://192.168.11.210:6000`；备用走 CAN：`--hand-port /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0` |
| Vision Pro | 头显上跑 Tracking Streamer，本机是客户端；IP 每次从界面上读 |
| 相机（只有录制时用） | 两台 Orbbec：顶部 Gemini 335（序列号 CP0BB53000FC）、腕部 Gemini 305（序列号 CV27864000V2） |
| 珞石 SDK | `数据_硬件/xCoreSDK-Python`，v0.7.1（servoJ 用的 `setServoJoint` 要这个版本） |

---

## 已知问题

- **解支跳变、换构型时臂会甩**：10-10 基线那次「解→解」最大 173.8°，10-15 那次 47.5°。读法见 [遥操作_servoJ](遥操作_servoJ.md) 第六节。
- **腕部 Gemini 305 不能走 OBSENSOR**：一帧不出，还会把 V4L2 也弄坏，只能物理拔插救回来。
  10-15 起 `atom_robot` 的 config 让它走 V4L2（只开相机核对过，servoJ 录制里还没真跑过），见 [数据采集_servoJ](数据采集_servoJ.md) 第 2 节。
  别同时开 RobotJev（它只走 OBSENSOR）。
- `--cam-top` / `--cam-front` 的帮助文字过时，还写着「序列号都一样」「不在腕上」。
- `--ee gripper` 只记录指令、不驱动夹爪；那只航凯夹爪 9/30 过载后卡滞，在等厂家。
- SO101 和 VR 两条线的采集默认都存到 `~/atom_episodes`，一个任务一个子目录，两条线的数据别混在一个目录里。
- C++ 笛卡尔后端的编译目录带着旧路径缓存，要删掉重编，见 [cartesian_rt/README.md](cartesian_rt/README.md)。

---

## 文档索引

| 文档 | 讲什么 |
|---|---|
| [遥操作_servoJ](遥操作_servoJ.md) | servoJ 怎么用、原理、参数、日志怎么看、出问题看哪里 |
| [数据采集_servoJ](数据采集_servoJ.md) | 在 servoJ 上录数据、检查、删改、转数据集 |
| [AR5遥操三条路](AR5遥操三条路.md) | 本地逆解 + 关节实时流、MoveL、servoJ 三条路是怎么试出来的 |
| [avp_arm_teleop操作手册](avp_arm_teleop操作手册.md) | 对照用的 MoveL / cart / rt 三个后端 |
| [cartesian_rt/README.md](cartesian_rt/README.md) | C++ 笛卡尔实时后端的编译和验证 |
| [调试历史_2026-09-20](调试历史_2026-09-20.md) | 修复前的调试记录，只作历史参考 |
| [工作区说明](../../../工作区说明.md) | 整个工作区的结构、环境、迁移、离线测试基准 |
| [数据转换说明](../../../算法/DexGraspVLA/数据转换说明.md) | 采集数据转成 DexGraspVLA 数据集 |
| [SO101 数据采集](../../SO101/数据采集.md) | SO-101 主从臂那条采集线（不是 Vision Pro） |
