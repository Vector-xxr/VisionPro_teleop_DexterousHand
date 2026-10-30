# VisionPro_teleop_DexterousHand

用 Apple Vision Pro 的手部追踪，遥操珞石 AR5 七轴机械臂和因时 RH56 灵巧手：
左手带着臂的末端和五根手指走，右手捏合当离合。臂走 servoJ，逆解在本地做（闭式解析解），可以边遥操边录数据。

所有代码和文档都在 [`VR设备/`](VR设备/) 里，**详细说明从 [`VR设备/README.md`](VR设备/README.md) 开始看**。

## 现状

- 主程序是 `VR设备/avp_servoj_teleop.py`（servoJ）。2026-10-10 一次 275 秒的真机任务：逆解失败 0/25589，端到端延迟 72 ms，对齐后残差 1.9 mm，手 20 Hz、0 错。
- 2026-10-15 AR5 驱动换成工作区里共用的 `atom_robot` 后，真机复验通过：逆解失败 0，端到端 72 ms，残差 1.5 mm。
- 边遥操边录数据（`--record`）的代码写好了，**还没在 servoJ 上正式采过数据**。

## 这个仓库不是自包含的

它是一个更大工作区里的一个目录（原位置 `<工作区>/数据_硬件/灵巧手遥操/`），连真机要用到仓库外的两样东西：

| 依赖 | 是什么 | 在原工作区的位置 |
|---|---|---|
| `atom_robot` | AR5 臂驱动（`AR5Env`）、配置、坐标变换、相机驱动。和 SO-101 主从臂采集共用一份 | `算法/DexGraspVLA-/SO101/atom_robot/`（另一个仓库，目前没有公开） |
| `xCoreSDK-Python` v0.7.1 | 珞石官方 Python SDK，servoJ 用的 `setServoJoint` 要这个版本 | `数据_硬件/xCoreSDK-Python/` |

代码靠 `VR设备/ws_paths.py` 找它们：从自身位置往上找「同时有 `算法/` 和 `数据_硬件/` 的目录」当工作区根。
单独克隆这个仓库时，按原工作区的目录结构把两样东西放好，或者用环境变量 `YUHANG_WS` 指到工作区根（SDK 位置也可以用 `XCORE_SDK` 单独指定）。

没有这两样时：
- **不能用**：遥操主程序（`avp_servoj_teleop.py`、`avp_arm_teleop.py`、`avp_movel_teleop.py`）、回位和救援脚本（`home_arm.py` 等），导入时就报 `No module named 'atom_robot'`。
- **能单独用**：灵巧手驱动 `inspire_hand6.py`、手部映射 `hand_retarget.py`、AR5 逆解（`ar5_poe_ik.py`、`ar5_ik_select.py`、`ar5_sew.py`、`paden_kahan.py`）、采集后处理（`qc_episodes.py`、`replay_episode.py`、`manage_episodes.py`、`to_lerobot_dex.py`）、假头显 `mock_avp.py`。

`VR设备/` 下文档里指向 `工作区说明.md`、`SO101`、`DexGraspVLA` 的链接，指的是原工作区里的其他项目，在这里打不开。

## 快速开始（在原工作区里）

```bash
cd <工作区>/数据_硬件/灵巧手遥操/VR设备
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python avp_servoj_teleop.py 192.168.x.x --planner spline --lag-ms 50 \
    --rot-mode beside --ws-max-z 0.60 --w-psi 0.5 --max-step-rad 0.015
```

- `192.168.x.x` 换成头显上 Tracking Streamer 界面显示的 IP。
- 一启动臂会先关节回位（3 秒内 Ctrl+C 可取消，`--no-home` 跳过）。回车挂离合，右手保持捏合臂才跟；`h` 回位后退出，`q` 退出。
- `PYTHONPATH=` 是因为原机器的 ROS 会污染 conda 环境；conda 环境 `dexgraspvla` 里要有 `pymodbus`（2.5.3）、`avp_stream`、`grpc`、`ruckig`。**`VR设备/requirements.txt` 已过时，别照它装。**
- 不连硬件的离线测试：`VR设备/` 下 `python test_changes.py`（要 `atom_robot`），最后打印「全部通过」。

## 硬件

| 设备 | 说明 |
|---|---|
| 珞石 AR5 | 七轴，控制器 `192.168.2.160`，本机网卡 `192.168.2.222` |
| 因时 RH56 灵巧手 | 默认走网口 Modbus TCP（`tcp://192.168.11.210:6000`），备用 CAN |
| Apple Vision Pro | 头显上跑 Tracking Streamer，本机是客户端 |
| 两台奥比中光 Orbbec（只在录数据时用） | 俯视 Gemini 335、腕部 Gemini 305 |

## 文档

| 文档 | 讲什么 |
|---|---|
| [`VR设备/README.md`](VR设备/README.md) | 入口：状态、启动、按键、回位救援、文件地图、已知问题 |
| [`VR设备/遥操作_servoJ.md`](VR设备/遥操作_servoJ.md) | servoJ 怎么用、原理、参数、日志怎么看 |
| [`VR设备/数据采集_servoJ.md`](VR设备/数据采集_servoJ.md) | 边遥操边录数据、检查、转数据集 |
| [`VR设备/AR5遥操三条路.md`](VR设备/AR5遥操三条路.md) | 本地逆解 + 关节实时流、MoveL、servoJ 三条路是怎么试出来的 |
| [`VR设备/avp_arm_teleop操作手册.md`](VR设备/avp_arm_teleop操作手册.md) | 对照用的 MoveL、cart、rt 三个后端 |
| [`VR设备/调试历史_2026-09-20.md`](VR设备/调试历史_2026-09-20.md) | 修复前的调试记录，只作历史参考 |

不在仓库里的：运行日志（`logs/`）、录的数据、C++ 后端的编译产物，以及早期 vuer 方案（`vision_pro_pose.py`）用的证书 `cert.pem` / `key.pem`（要用时 `VR设备/gen_cert.py` 重新生成；servoJ 主程序用不到）。
