# Vision Pro servoJ 数据采集

遥操怎么用见 [遥操作_servoJ](遥操作_servoJ.md)。这份是**在 servoJ 遥操上面采数据**：怎么录、怎么检查、怎么删、怎么转成数据集。
SO-101 主从臂那条采集线是另一套，见 [SO101 数据采集](../../SO101/数据采集.md)。

> **这条路还没真正采过数据。** 本机 `~/atom_episodes`、`~/vel_episodes` 里的 497 个 npz 都没有 `ee_target` 字段，
> 而这是 `episode_recorder.py` 必写的，所以都不是 servoJ `--record` 录的。下面的键位和参数对照代码核过，
> 但第一次真机采的时候要盯着帧率和存盘提示。

一条 episode = 一个 `.npz` + 每路相机一个 `.mp4`。装灵巧手时数据是 13 维：AR5 七个关节角（弧度）加因时手六根手指
（0~1000，0 是握紧）。动作空间是**绝对关节角**，不是增量。

```
1 环境  →  2 检查相机  →  3 采集  →  4 回放 + 自动检查  →  5 删和重排  →  6 转数据集
```

---

## 1. 环境

采集、检查、回放、删改都在 `VR设备` 目录下，用 dexgraspvla 环境：

```bash
cd ~/yuhang_workspace/数据_硬件/灵巧手遥操/VR设备
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python qc_episodes.py --help
```

转 lerobot 数据集那一步换 `lerobot` 环境（见第 6 节）。dexgraspvla 的 torch 版本是配死的，装 lerobot 会动到它。

**SO101 和 VR 两条线默认都存到 `~/atom_episodes`**，两边的 npz 格式不一样。一个任务一个子目录（`--out-dir ~/atom_episodes/任务名`），
两条线的数据不要放进同一个目录。本目录的工具碰到 SO101 那种格式会逐条跳过并说明原因，不会崩。

---

## 2. 检查相机

每次开工先看一眼。相机没插好、镜头转了、画面冻了，这些在数据里都是安静的。

现在是两台 Orbbec，键名沿用 `atom_robot/ar5/config.py` 的 `CameraConfig.sources`：

| 键名 | 相机 | 位置 | config 里的写法 |
|---|---|---|---|
| `cam_high` | Gemini 335 | 顶部，俯视工作区（9/29 换上来的，之前是绿联） | `orbbec:CP0BB53000FC`（按序列号，走 OBSENSOR） |
| `cam_front` | Gemini 305 | **腕部**，跟着臂走（9/22 起） | `v4l2:/dev/ar5_cam_wrist`（10-15 起走 V4L2，见下） |

`cam_front` 这个名字是历史原因（原来是台架侧视相机），**9/22 之后它是腕部视角**；`cam_high` 9/29 前后也不是同一台相机。

**腕部 305 不能走 OBSENSOR**：会一帧不出，还会把 V4L2 也弄坏，只能物理拔插救回来。所以 10-15 起 config 里它写成 `v4l2:/dev/ar5_cam_wrist`，明确走 V4L2
（当天真机核对：335 走 OBSENSOR、305 走 V4L2 同时开，29.7 / 30.1 Hz，640×480）。录制和看相机都按 config 来，不用另外给参数。

- 看相机用 SO101 的 `atom_robot.ar5.view_camera`，**不要用本目录的 `view_cameras.py`**（它用 V4L2 去开所有相机，335 被 V4L2 碰过之后 OBSENSOR 会掉速）。
- 别让任何程序用 OBSENSOR 开 305，包括 RobotJev（它的相机代码只走 OBSENSOR）。

```bash
cd ~/yuhang_workspace/数据_硬件/SO101
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python -m atom_robot.ar5.view_camera --list                           # 列设备，不打开 Orbbec
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python -m atom_robot.ar5.view_camera --name cam_high   # 顶部 335，走 OBSENSOR
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python -m atom_robot.ar5.view_camera                    # 默认看 cam_front：腕部 305，走 V4L2
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python -m atom_robot.ar5.view_camera --all              # 两路并排
```

- `/dev/ar5_cam_wrist` 是 udev 规则给的固定名（规则 `数据_硬件/SO101/atom_robot/ar5/99-ar5-cameras.rules`，安装见 [工作区说明](../../../工作区说明.md) 6.4）。
- 窗口里 `q` 退出、`g` 叠九宫格摆正相机、`s` 存图、`i` 叠信息条。没有图形界面时加 `--save-only 30` 只抓帧存图。
- 同一台 Orbbec 别在 OBSENSOR 和 V4L2 之间来回切：V4L2 用过之后 OBSENSOR 会掉到约 0.3 Hz（驱动发现帧率低会自动 USB 复位、等 8 秒再开）；
  OBSENSOR 用过之后 V4L2 一帧不出，只能物理拔插。

**要确认三件事**：两路画面都出得来；桌面和物体都在画面里；两路各是什么视角。

---

## 3. 采集

在 `VR设备` 目录下，主命令（见 [遥操作_servoJ](遥操作_servoJ.md) 第一节）后面加录制参数：

```bash
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python avp_servoj_teleop.py 192.168.x.x --planner spline --lag-ms 50 \
    --rot-mode beside --ws-max-z 0.60 --w-psi 0.5 --max-step-rad 0.015 \
    --record --out-dir ~/atom_episodes/pick_bolt --task "pick up the bolt and put it in the box" \
    --num-episodes 50 --frame-cap 900 --reset-seconds 5
```

- `192.168.x.x` 换成头显界面上显示的 IP；`pick_bolt` 和 `--task` 换成这批的任务。
- 相机按 config 开：335 走 OBSENSOR，305 走 V4L2（第 2 节）。这个组合 10-15 只开相机核对过，**还没在 servoJ 录制里真跑过**。
- 一启动臂会先关节回位（`--no-home` 跳过）。

### 键盘

| 键 | 做什么 |
|---|---|
| **回车** | 挂 / 摘离合。挂上之后**右手保持捏合**臂才动，松开即停 |
| **空格** | 开始录这一条。**要先挂上离合、右手捏住**，否则不响应 |
| **s** | 存盘，**标成功** |
| **←** | 存盘，**不标成功**（不删，只是不当作成功样本） |
| **退格** | 丢弃这一条，编号不变，重录 |
| **h** | 停下、关节回位、程序退出 |
| **q** | 退出 |

- 成功与否写进 npz 的 `success` 字段，训练时按它筛，不用靠人记。按 ← 保留下来的条目仍然是有效数据 ——
  失败轨迹对某些训练方法有用，而且当场判不准的时候留着比删了强。
- 录到 `--frame-cap` 帧会自动存盘并**标成功**。
- 录着的时候按 `q` / `h` / Ctrl+C 退出，这一条会存下来但**标成不成功**。

**存完不会自动开下一条。** 这是故意的：自动往下开的话，每条的头几秒都是你在摆现场，
数据看着正常但前面是空的。`--reset-seconds` 在存盘后挡住空格若干秒，防止手快。

### 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--record` | 关 | 开启采集。不开就不碰相机 |
| `--task` | 空 | 这批数据在做什么，一句英文小写祈使句。**一批之内不要改**，改了不同 episode 的语言条件就不一致。不给会警告 |
| `--out-dir` | `~/atom_episodes` | 存哪。**一个任务一个子目录** |
| `--num-episodes` | 0（不限） | 采够多少条自动收工 |
| `--frame-cap` | 0（不限） | 一条最多多少帧，到了自动存盘。**按帧数不按秒**：回路周期抖一下不该改变一条的长短。30 Hz 下 900 帧 = 30 秒 |
| `--reset-seconds` | 5 | 存完一条之后空格不响应的秒数 |
| `--record-hz` | 30 | 录制帧率。相机就是 30 fps，设高了只会录到重复帧 |
| `--cam-top` / `--cam-front` | 用 config 里的 | 覆盖 `cam_high` / `cam_front` 的源，写法同 config（`orbbec:<序列号>`、`v4l2:<路径>`）。305 别写成 `orbbec:…` |
| `--ee` | `hand` | 末端：`hand` 因时手 6 维；`gripper` 夹爪 1 维，**只记录左手捏合算出的指令、不驱动夹爪**；`none` 只有臂 |
| `--auto-record` | 关 | 接合即开录、摘离合即存盘。**别用**，理由见上 |

编号从目录里已有的最大值往后接，不会覆盖。

### 采的时候注意

* **每条都要换位置。** 物体摆在同一个点采 100 条，训出来的策略只会走那一条轨迹。
  自动检查第 10 项会量起始末端位置的离散度。
* **手要真的张合。** 全程握着不动的话，手那六维学不到东西，自动检查第 9 项会报出来。
* **一条别太长。** 30 秒够做一次抓放。太长的条目里大半是空转。
* **开始录之前先把手放到起始位置**，不要按了空格再去找物体。
* 存盘时会当场打 ⚠ 提醒：太短、末端几乎没动、末端走得太少、实测帧率偏离、相机少给了帧。

---

## 4. 回放和自动检查

### 自动检查

```bash
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python qc_episodes.py --src ~/atom_episodes/pick_bolt
```

查十项，逐项打印数字，最后列出要处理的问题：

| # | 查什么 | 判据 |
|---|---|---|
| 1 | 丢帧 | npz 帧数和每路 mp4 帧数一致 |
| 2 | 重复帧 | 相邻两帧逐像素完全相同 = 同一张图被读了两次 |
| 3 | 画面丢失 | 黑帧、低对比帧 |
| 4 | 帧率 | 实测对标称，逐帧间隔的 p50 / p99 / 最大 |
| 5 | 卡顿 | 连续 5 帧几乎不变 |
| 6 | 相机漂移 | 首帧相对静态背景的位移，超过 `--thresh`（默认 8 px）报警 |
| 7 | 人手误入 | 肤色占比。**这条不可靠**，木桌和黄胶带都会撞色 |
| 8 | 有效动作 | 开头空转 / 结尾空转 / 中间停顿各占多少 |
| 9 | 末端行程 | 每条内部张合够不够、有没有真正合拢过、有没有哪条全程没动 |
| 10 | 场景多样性 | 起始末端位置的标准差和跨度 |

第 6 项只在**跨 episode 不变的像素**上量。整帧比对会被臂的起始位置和物体的摆位带偏 ——
以前实测过一条整帧报 73 px、其实相机根本没动。静态区不写死坐标，从所有首帧的逐像素
标准差里自己算，换机位也不用改代码。腕部相机跟着臂走，这一项对 `cam_front` 参考意义不大。

第 9 项**不要拿「有没有用满 0~1000」当判据**。任务用不到量程顶部不是缺陷，归一化会自己
适配。要查的是每条内部的行程、有没有合拢到底、有没有哪条手全程没动。

**有两件它查不出来，只能用眼睛看**：人手有没有入镜，以及操作对不对（抓稳没有、放进去
没有）。后者没有自动判据。

### 回放

```bash
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python replay_episode.py --src ~/atom_episodes/pick_bolt                         # 列出全部，一行一条
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python replay_episode.py --src ~/atom_episodes/pick_bolt --episode 7             # 看第 7 条
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python replay_episode.py --src ~/atom_episodes/pick_bolt --all --only-flagged    # 只看列表里带 ⚠ 的
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python replay_episode.py --src ~/atom_episodes/pick_bolt --episode 7 --cam cam_front   # 只看腕部那一路
```

`--episode` 给的是**文件名里的编号**，不是列表下标。删过数据之后两者会对不上，
所以按编号找不到时会把现有编号列出来。`--cam` 给键名（`cam_high` / `cam_front`）。

窗口里：空格暂停、`←` `→` 单帧步进、`a` `d` 上一条下一条、`q` 退出。
底部的绿色条是末端开合（因时手六条），顶部一行是这一帧末端走了多少毫米。

---

## 5. 删数据、编号、文件名

```bash
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python manage_episodes.py --src ~/atom_episodes/pick_bolt                        # 列出现状：编号、空洞、孤儿文件
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python manage_episodes.py --src ~/atom_episodes/pick_bolt --delete 3,7,12        # 先预览
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python manage_episodes.py --src ~/atom_episodes/pick_bolt --delete 3,7,12 --apply
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python manage_episodes.py --src ~/atom_episodes/pick_bolt --clean-orphans --apply
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python manage_episodes.py --src ~/atom_episodes/pick_bolt --renumber --apply     # 补空洞（一批彻底做完再用）
```

**用这个删，不要手动 `rm`。** 一条 episode 是一个 npz 加两个 mp4，手动删最容易只删掉
npz，留下的 mp4 会让后面的检查和转换对不上。`--clean-orphans` 就是收拾这种残局的。

### 编号要不要重排

**采集过程中不要排。** 三条理由：

* 有空洞不影响任何下游。`to_lerobot_dex.py` 按文件名排序读，lerobot 自己重新编
  `episode_index`，空洞会被自动吃掉。
* 重排会让你手上的记录失效。回放时记下「第 12 条抓歪了」，一重排 12 号就是另一条了。
  转 DexGraspVLA 用的 `head_bbox.json` 也是按编号记的（带帧数指纹，对不上会拒转）。
* 重排是批量改名，中途断掉就是一地半新半旧的文件名。

要排就等这一批采完、检查完、转完数据集之后再排一次。`--renumber` 默认只预览，带
`--apply` 才动，并且分两步改名（先全部改成临时名再落位），避免 `1 -> 0` 这种撞车。
映射表写进 `renumber_log.txt`，回放笔记按它对一遍。

### 文件名要不要改

**不用改。** 现在的 `episode_%04d.npz` + `episode_%04d_<相机名>.mp4` 已经够用：编号定位、
相机名在后缀。任务描述、成功标记、时间戳、实测帧率都在 npz 里，`manage_episodes.py`
不带参数就能列出来。**一个任务一个 `--out-dir`**，这比在文件名里区分任务干净得多。

---

## 6. 转成数据集

### lerobot

```bash
cd ~/yuhang_workspace/数据_硬件/灵巧手遥操/VR设备
PYTHONPATH= ~/miniforge3/envs/lerobot/bin/python to_lerobot_dex.py --src ~/atom_episodes/pick_bolt --repo-id atom_ar5_bolt
```

时间戳用 npz 里的 `fps_measured`，不是标称的 30 —— 回路实际跑多快事后只能靠它。

### DexGraspVLA

用 `to_dexgraspvla.py`（dexgraspvla 环境）。servoJ 存的 `action` 本来就是绝对关节角，用默认的 `--action stored`；
13 维（7 臂 + 6 手）和 DexGraspVLA 默认的维度一致。正式数据要用 `--mask sam`，每条先要有头部相机第 0 帧上的目标框：

```bash
cd ~/yuhang_workspace/数据_硬件/灵巧手遥操/VR设备
# 先补框：只对还没有框的 episode 弹窗手画，写进 <src>/head_bbox.json
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python to_dexgraspvla.py --src ~/atom_episodes/pick_bolt --bbox-only \
    --wrist-cam cam_front --head-cam cam_high
# 再转换
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python to_dexgraspvla.py --src ~/atom_episodes/pick_bolt \
    --dst ~/pick_bolt.zarr --wrist-cam cam_front --head-cam cam_high --mask sam
```

- 格式、框、掩码、检查的细节见 [数据转换说明](../../../算法/DexGraspVLA-/数据转换说明.md)。
- `算法/DexGraspVLA-/make_h100_dataset.sh` 现在**只支持 SO101 的 8 维数据**（写死了 `--action next-state` 和 `--dim 8`），servoJ 的数据别用它。
- `to_dexgraspvla.py` 被 `make_h100_dataset.sh` 按相对路径调用，**不能挪位置、不能改名**。

---

## 附：npz 里有什么

| 字段 | 形状 | 内容 |
|---|---|---|
| `state` | (n, 7+末端维数) | 实测：7 个关节角（rad）+ 末端实测。因时手 6 维（0~1000，0 握紧），夹爪 1 维（0~1，0 合拢），`--ee none` 没有 |
| `action` | 同上 | 指令：servoJ 下发的绝对关节角 + 末端的绝对指令 |
| `ee_pose` | (n, 6) | 实测末端位姿：位置（m）+ 轴角（rad） |
| `ee_target` | (n, 6) | 指令末端位姿。留着以后换动作空间时重算 |
| `timestamp` | (n,) | 每帧相对第一帧的秒数，实测。**转数据集用它** |
| `success` | 标量 | 按 s 或到帧数上限存的是 True；按 ←、或录着时退出存的是 False |
| `ee_connected` | 标量 | 末端硬件在不在。False 时 `state` 的末端维是拿指令顶的，不是实测 |
| `task` / `fps` / `fps_measured` / `camera_keys` / `arm_dof` / `ee_dof` / `ee_names` / `ee_range` | | |

写盘在后台线程上。servoJ 要求每 8 ms 收到一个关节角，主循环停 2 秒就会被控制器判成
通信丢包，而两路相机 600 帧写一次实测要 2 秒出头。

---

## 附：以前踩过的坑

* **相机按 `videoN` 认** —— 拔插后编号互换，视角对调，数据全废。现在两台 Orbbec 按序列号认。
* **一条的长度按秒切** —— 回路周期一抖，同样是 30 秒的两条帧数差一截。按帧数切。
* **存完自动开下一条** —— 每条开头几秒都是操作员在摆现场。存完停下，人按空格再开。
* **拿「末端用没用满量程」当判据** —— 误报。任务用不到量程顶部不是缺陷。
* **整帧比对量相机漂移** —— 被臂的起始位置带偏，报 73 px 实际没动。只在静态像素上量。
* **肤色检测判人手误入** —— 和木桌、黄胶带撞色，报出来的基本都是误报。只能回放确认。
* **手动 `rm` 删数据** —— 留下孤儿 mp4，后面的检查和转换对不上。
* **一批采到一半改 `--task` 或 `--record-hz`** —— 不同 episode 的尺度和语言条件不一致。

---

## 文件清单

| 文件 | 作用 |
|---|---|
| `avp_servoj_teleop.py` | 遥操主程序，`--record` 开启采集 |
| `episode_recorder.py` | 一条 episode 的内存缓冲 + 后台写盘 |
| `qc_episodes.py` | 采完一批的自动检查，十项 |
| `replay_episode.py` | 按编号回放，用眼睛确认 |
| `manage_episodes.py` | 删条目、清孤儿、补编号 |
| `to_lerobot_dex.py` | 转 lerobot 数据集（在 lerobot 环境里跑） |
| `to_dexgraspvla.py` | 转 DexGraspVLA 的 zarr 数据集（dexgraspvla 环境） |
| `数据_硬件/SO101/atom_robot/ar5/view_camera.py` | 开录前看相机（在 SO101 目录下 `-m` 运行） |
