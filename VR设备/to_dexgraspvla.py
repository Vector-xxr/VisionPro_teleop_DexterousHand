#!/usr/bin/env python3
"""把采到的 episode 转成 **DexGraspVLA** 的 zarr 数据集。

    python to_dexgraspvla.py --src ~/atom_episodes/<任务> --dst <数据集>/<任务>.zarr \
        --wrist-cam cam_front --head-cam cam_high --action next-state --mask sam

一般不直接调它，用 `算法/DexGraspVLA/make_h100_dataset.sh`（参数、环境、校验、
校验和都包好了）。

格式是从 `算法/DexGraspVLA/DexGraspVLA` 的代码里读出来的，不是猜的：
`controller/dataset/mask_image_dataset.py` 只读四个 key，
`controller/config/task/*.yaml` 给了形状，`inference.py` 给了数值约定。

    <dst>.zarr/
    ├── data/
    │   ├── right_cam_img  (N, H, W, 3) uint8    腕部相机 BGR
    │   ├── rgbm           (N, H, W, 4) uint8    头部相机 BGR + 第 4 通道 0/1 掩码
    │   ├── right_state    (N, D) float32        实测
    │   └── action         (N, D) float32        指令
    └── meta/
        └── episode_ends   (E,) int64            每条的**累计**结束下标

所有 episode 沿 0 轴首尾相接，`episode_ends` 是切点，
`data[key].shape[0] == episode_ends[-1]` 是它自己会断言的不变量。
图像存 HWC、原分辨率（640×480），loader 自己转 CHW 并缩放到 518×518。
根组的 attrs 记了关节限位、相机、来源 episode 和框 —— 推理时把动作
还原成弧度要用同一套限位，别丢。

**数值约定**（`inference.py:657` 的 `get_state` 和 `:760` 的执行）：

* 前 7 维是臂关节，**归一化到 [-1, 1]**，用各自的关节限位缩放。
  推理时再 `scale()` 回弧度，所以存的是无量纲值。
  这里用的是**我们这台 AR5 的软限位**，不是他们配置里那台臂的。
* 后面是手/夹爪，**原始归一化值 0~1**，1 = 全开（他们的 `default_open` 全是 1.0）。
* **动作是绝对值，不是增量。** servoJ 那条路（绝对关节角）默认就能转；
  主从臂 VEL 那条路的 `action` 栏是位姿增量用不了，但**七个关节角在 `state` 里**，
  所以加 `--action next-state` 也能转 —— 拿下一帧的实测关节角当动作。
  不给这个参数会拒绝，不会悄悄转出一份错的。

**维度**：他们的 `shape_meta` 写死 13 = 7 臂 + 6 手，正好对上 AR5 + 因时手。
换成一维夹爪就是 8，训练时用命令行覆盖 `shape_meta` 两行，官方 ckpt 走部分加载
（`+training.start_ckpt_skip_mismatch=true`，两端 4 个张量随机初始化）。
脚本结束时会把该加的参数打出来。

**mask 那一路**：第 4 通道是目标物体的二值掩码（0/1，和官方数据一样），
是控制器知道「抓哪个」的唯一输入。正式数据用 `--mask sam`：

    --mask sam --bbox-file <json>       # 默认 <src>/head_bbox.json
                                        # {"episode_0000": {"box": [x1, y1, x2, y2], "frames": 700}, ...}
                                        # 头部相机第 0 帧上的目标框，640×480 像素坐标；
                                        # frames 是这条的帧数，对不上就拒转（编号重排过，
                                        # 框会套到别的 episode 上）。也接受旧写法 [x1, y1, x2, y2]
    --select-bbox                       # 没框的 episode 弹窗手画（cv2.selectROI），边画边存
    --bbox-only                         # 只补框不转换：python to_dexgraspvla.py --src <任务目录> --bbox-only

流程和 `DexGraspVLA/inference.py` 的 initialize_sam_cutie / get_mask 一致：
第 0 帧 SAM vit_h（box 提示、multimask 取最高分）→ Cutie 逐帧跟踪，输入都转 RGB。
每条存一张质检图（首/中/末帧叠掩码）到 `<dst>_mask_qc/`，并报告掩码为空（跟丢）的帧数。
`--mask none` 全填 255 只能验链路，**训不出对的东西**。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ARM_DOF = 7


def read_video(path, n_expect):
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        # 存 BGR，原样不转。官方数据就是 BGR（obs_encoder.py:148/176 进模型前
        # 自己翻成 RGB，inference.py 也处处 [..., ::-1]）；按 RGB 解读官方
        # tiny dataset 木地板会变蓝。这里以前转成了 RGB，进 DINOv2 就是颜色反的。
        frames.append(f)
    cap.release()
    if len(frames) != n_expect:
        print(f"    ⚠ {path.name}: {len(frames)} 帧，npz 说 {n_expect} 帧")
    return frames


class SamCutieMasker:
    """第 0 帧 SAM 出掩码，Cutie 往后跟。和 inference.py 同一套调用方式。"""

    def __init__(self, sam_ckpt):
        import torch
        from segment_anything import sam_model_registry, SamPredictor
        from cutie.utils.get_default_model import get_default_model
        self.torch = torch
        sam = sam_model_registry["vit_h"](checkpoint=str(sam_ckpt)).cuda()
        self.predictor = SamPredictor(sam)
        self.cutie = get_default_model()

    def __call__(self, frames_bgr, bbox):
        """frames_bgr: list of (H,W,3) uint8 BGR。返回 (T,H,W) uint8 0/1 掩码和 SAM 分数。"""
        from cutie.inference.inference_core import InferenceCore
        from torchvision.transforms.functional import to_tensor
        torch = self.torch
        self.predictor.set_image(frames_bgr[0][..., ::-1])            # SAM 要 RGB
        masks, scores, _ = self.predictor.predict(box=np.asarray(bbox, dtype=float),
                                                  multimask_output=True)
        m0 = masks[np.argmax(scores)].astype(np.uint8)
        # 每条新建一个 InferenceCore：比 inference.py 的 clear_memory() 更干净，
        # clear_memory 不清 object_manager
        proc = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        proc.max_internal_size = -1                                    # 原分辨率跟踪
        out = np.empty((len(frames_bgr),) + m0.shape, dtype=np.uint8)
        with torch.no_grad():
            for i, f in enumerate(frames_bgr):
                img = to_tensor(f[..., ::-1].copy()).cuda().float()
                if i == 0:
                    prob = proc.step(img, torch.from_numpy(m0).cuda(), objects=[1])
                else:
                    prob = proc.step(img)
                out[i] = proc.output_prob_to_mask(prob).cpu().numpy().astype(np.uint8)
        return out, float(np.max(scores))


def save_mask_qc(path, frames_bgr, masks, bbox):
    """首/中/末三帧叠掩码拼一张，第 0 帧画上框。"""
    import cv2
    tiles = []
    for i in (0, len(frames_bgr) // 2, len(frames_bgr) - 1):
        im = frames_bgr[i].copy()
        m = masks[i].astype(bool)
        im[m] = (0.45 * im[m] + 0.55 * np.array([255, 120, 0])).astype(np.uint8)
        if i == 0:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(im, f"t={i} area={m.mean():.3f}", (8, 24), 0, 0.7, (0, 255, 255), 2)
        tiles.append(im)
    cv2.imwrite(str(path), cv2.resize(np.hstack(tiles), None, fx=0.5, fy=0.5))


def select_bboxes(eps, head, bboxes, bbox_file):
    """给还没有框的 episode 弹窗手画。回车/空格确认，c 跳过这条。每画一条就落盘。"""
    import json
    import cv2
    for p in eps:
        if p.stem in bboxes:
            continue
        cap = cv2.VideoCapture(str(p.with_name(f"{p.stem}_{head}.mp4")))
        ok, f0 = cap.read()
        cap.release()
        if not ok:
            continue
        x, y, w, h = cv2.selectROI(f"{p.stem}  框目标物体，回车确认，c 跳过", f0, showCrosshair=True)
        cv2.destroyAllWindows()
        if w > 0 and h > 0:
            bboxes[p.stem] = {"box": [int(x), int(y), int(x + w), int(y + h)],
                              "frames": int(len(np.load(p)["state"]))}
            bbox_file.write_text(json.dumps(bboxes, indent=1, ensure_ascii=False))
            print(f"  {p.stem} 框 {bboxes[p.stem]['box']}")


def bbox_of(entry, npz_path):
    """框文件的一条 → [x1, y1, x2, y2]。带 frames 的要和这条的帧数对上。

    框按文件名对应 episode。`manage_episodes.py --renumber` 重排过编号、而框文件
    没跟着改，框就会套到别的 episode 上 —— SAM 照样分割出一个东西，掩码看起来
    正常，只是抓的不是那个物体。帧数是最便宜的指纹，对不上就停。
    """
    if isinstance(entry, dict):
        n = int(len(np.load(npz_path)["state"]))
        if "frames" in entry and int(entry["frames"]) != n:
            raise ValueError(f"{npz_path.stem}：框文件记的是 {entry['frames']} 帧，"
                             f"这条实际 {n} 帧 —— 编号重排过？重新框这条")
        entry = entry["box"]
    return [int(v) for v in entry]


def normalise_arm(q, lo, hi):
    """弧度 → [-1, 1]。和 inference.py 的 unscale() 同一套。"""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    mid, half = (hi + lo) / 2.0, (hi - lo) / 2.0
    return np.clip((np.asarray(q, float) - mid) / np.maximum(half, 1e-9), -1.0, 1.0)


def main() -> int:
    import os
    from ws_paths import DEXGRASPVLA, SO101
    # Cutie 要的 resnet50 用 DexGraspVLA 项目里的 weights/torch_home（拷走工作区就能断网跑）
    os.environ.setdefault("TORCH_HOME", str(DEXGRASPVLA / "weights" / "torch_home"))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="~/atom_episodes", help="采集输出目录")
    ap.add_argument("--dst", default="", help="输出的 .zarr 路径（--bbox-only 时不用给）")
    ap.add_argument("--wrist-cam", default="", help="哪一路当 right_cam_img，留空取第一路")
    ap.add_argument("--head-cam", default="", help="哪一路当 rgbm，留空取第二路")
    ap.add_argument("--mask", choices=["none", "alpha", "sam"], default="none",
                   help="第 4 通道怎么来。sam=第 0 帧框 + SAM + Cutie 跟踪（正式数据用这个）；"
                        "none=全 255（占位，训不出东西）；"
                        "alpha=头部相机视频若是 4 通道就取它的 alpha")
    ap.add_argument("--bbox-file", default="",
                   help="--mask sam 用：{episode 名: [x1,y1,x2,y2]}，头部第 0 帧像素坐标。"
                        "默认 <src>/head_bbox.json")
    ap.add_argument("--select-bbox", action="store_true",
                   help="没框的 episode 弹窗手画，画完写回 --bbox-file")
    ap.add_argument("--bbox-only", action="store_true",
                   help="只弹窗补框，补完就退出，不转换（隐含 --mask sam --select-bbox）")
    ap.add_argument("--sam-ckpt",
                   default=str(DEXGRASPVLA / "weights" / "sam_vit_h_4b8939.pth"))
    ap.add_argument("--episodes", default="",
                   help="只转这几条，逗号分隔的编号，如 0,3,12（默认全部）")
    ap.add_argument("--chunk", type=int, default=2,
                   help="zarr 分块的帧数。训练是逐帧随机读，读 1 帧要解压整块；"
                        "官方数据 rgbm 2 帧/块、腕部 3 帧/块。以前默认 64，数据加载会比计算还慢")
    ap.add_argument("--action", choices=["stored", "next-state"], default="stored",
                   help="动作从哪来。stored=用 npz 里的 action（要求它本来就是绝对关节角，"
                        "servoJ 那条路）；next-state=**拿下一帧的实测关节角当动作**，"
                        "主从臂那条路要用这个 —— 它存的 action 是位姿增量，直接用不了")
    a = ap.parse_args()
    if a.bbox_only:
        a.mask, a.select_bbox = "sam", True
    elif not a.dst:
        ap.error("要给 --dst（只补框用 --bbox-only）")

    try:
        import zarr
        from numcodecs import Blosc
    except ImportError:
        print("  没装 zarr。用 dexgraspvla 环境跑。")
        return 1
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from episode_recorder import read_layout, LayoutError

    src = Path(a.src).expanduser()
    eps = sorted(src.glob("episode_*.npz"))
    if not eps:
        print(f"  {src} 里没有 episode_*.npz")
        return 1

    # 关节限位取我们这台臂的，和采集用的是同一份配置（atom_robot/ar5/config.py）
    sys.path.insert(0, str(SO101))  # atom_robot 所在目录
    from atom_robot.ar5.config import AR5Config
    arm = AR5Config().arm
    jlo, jhi = np.asarray(arm.joint_min, float), np.asarray(arm.joint_max, float)

    d0 = np.load(eps[0], allow_pickle=True)
    legacy = False
    try:
        arm_dof, ee_dof, ee_names, (ee_lo, ee_hi), ee_conn = read_layout(d0)
    except LayoutError:
        # 主从臂那条路：state 是 7 关节角 + 夹爪，action 是位姿增量。
        # 关节角本身在 state 里，所以**能转** —— 只是动作要重新构造，见 --action。
        st, ac = np.shape(d0["state"]), np.shape(d0["action"])
        if len(st) == 2 and st[1] == 8 and len(ac) == 2 and ac[1] == 7:
            legacy, arm_dof, ee_dof = True, 7, 1
            ee_names, ee_lo, ee_hi, ee_conn = ["gripper"], 0.0, 1.0, True
            if a.action != "next-state":
                print(f"  ✗ 这是主从臂（atom_robot/ar5/collect.py）采的数据：state 是 7 关节角 + 夹爪，"
                      f"而 **action 是位姿增量**，不是绝对关节角。")
                print(f"    加 --action next-state：拿下一帧的实测关节角当动作。")
                print(f"    代价是它是**实测值不是指令值**，中间隔着一层跟踪误差；"
                      f"好处是关节角本来就在数据里，不用重做逆解。")
                return 2
            print(f"  主从臂格式：action 用「下一帧的实测关节角」构造")
        else:
            print(f"  ✗ {eps[0].name} 的形状 state{st} action{ac} 认不出来")
            return 2
    cams = [str(x) for x in d0["camera_keys"]]
    if len(cams) < 2:
        print(f"  ✗ 只有 {len(cams)} 路相机，DexGraspVLA 要两路（腕部 + 头部）")
        return 2
    wrist = a.wrist_cam or cams[0]
    head = a.head_cam or (cams[1] if cams[1] != wrist else cams[0])
    for name, key in (("--wrist-cam", wrist), ("--head-cam", head)):
        if key not in cams:
            print(f"  ✗ {name} {key} 不在数据里，有的是 {cams}")
            return 2
    dim = arm_dof + ee_dof
    print(f"  {len(eps)} 条   维度 {dim} = {arm_dof} 臂 + {ee_dof} 末端 {ee_names}")
    print(f"  腕部 right_cam_img ← {wrist}     头部 rgbm ← {head}")
    if not ee_conn:
        print(f"  ⚠ 末端没有位置反馈，right_state 的末端维等于 action，不是独立观测")

    if a.episodes:
        want = {int(x) for x in a.episodes.split(",") if x.strip()}
        eps = [p for p in eps if int(p.stem.split("_")[-1]) in want]
        print(f"  只转 {len(eps)} 条：{[p.stem for p in eps]}")

    masker, bboxes, empty_report = None, {}, []
    if a.mask == "sam":
        import json
        bbox_file = Path(a.bbox_file).expanduser() if a.bbox_file else src / "head_bbox.json"
        if bbox_file.exists():
            bboxes = json.loads(bbox_file.read_text())
        if a.select_bbox:
            select_bboxes(eps, head, bboxes, bbox_file)
        missing = [p.stem for p in eps if p.stem not in bboxes]
        if a.bbox_only:
            print(f"  {bbox_file}：{len(eps) - len(missing)}/{len(eps)} 条有框"
                  + (f"，还缺 {missing}" if missing else ""))
            return 0 if not missing else 2
        if missing:
            print(f"  ✗ {len(missing)} 条没有目标框（{bbox_file}）：{missing[:8]}"
                  f"{' …' if len(missing) > 8 else ''}")
            print(f"    加 --select-bbox 手画，或用 --episodes 只转有框的")
            return 2
        # 先把所有框核一遍再开始 —— 别转了半小时才在第 40 条上发现对不上
        try:
            boxes = {p.stem: bbox_of(bboxes[p.stem], p) for p in eps}
        except ValueError as e:
            print(f"  ✗ {e}")
            return 2
        sam_ckpt = Path(a.sam_ckpt).expanduser()
        if not sam_ckpt.exists():
            print(f"  ✗ SAM 权重不在 {sam_ckpt}")
            return 2
        print(f"  掩码：SAM vit_h + Cutie，框来自 {bbox_file}")
        masker = SamCutieMasker(sam_ckpt)

    dst = Path(a.dst).expanduser()
    if dst.exists():
        print(f"  ✗ {dst} 已存在，先删或换个名字")
        return 2
    root = zarr.open(str(dst), mode="w")
    data, meta = root.require_group("data"), root.require_group("meta")
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    arrays, ends, total, done, tasks = {}, [], 0, [], set()

    for p in eps:
        d = np.load(p, allow_pickle=True)
        if legacy:
            ad, ed, elo, ehi = 7, 1, 0.0, 1.0
        else:
            try:
                ad, ed, _n, (elo, ehi), _c = read_layout(d)
            except LayoutError as e:
                print(f"  跳过 {p.name}：{e}")
                continue
        if ad + ed != dim:
            print(f"  跳过 {p.name}：维度 {ad}+{ed} 和第一条的 {dim} 对不上")
            continue
        n = len(d["state"])
        vids = {k: read_video(p.with_name(f"{p.stem}_{k}.mp4"), n) for k in (wrist, head)}
        m = min([n] + [len(v) for v in vids.values()])
        if m == 0:
            print(f"  跳过 {p.name}：没有视频")
            continue

        # 臂归一化到 [-1,1]，末端归一化到 0~1
        def pack(arr):
            q = normalise_arm(arr[:m, :ad], jlo[:ad], jhi[:ad])
            e = ((arr[:m, ad:] - elo) / max(ehi - elo, 1e-9)) if ed else np.zeros((m, 0))
            return np.concatenate([q, np.clip(e, 0.0, 1.0)], axis=1).astype(np.float32)

        state_raw = np.asarray(d["state"], float)
        if a.action == "next-state":
            # 动作 = 下一帧的实测状态，最后一帧原地重复。
            # 这是「绝对位置动作空间」的标准重建法：策略学的是「下一步该到哪」。
            act_raw = np.concatenate([state_raw[1:], state_raw[-1:]], axis=0)
        else:
            act_raw = np.asarray(d["action"], float)
        chunk = {
            "right_state": pack(state_raw),
            "action":      pack(act_raw),
            "right_cam_img": np.stack(vids[wrist][:m]).astype(np.uint8),
        }
        rgb = np.stack(vids[head][:m]).astype(np.uint8)
        if masker is not None:
            masks, score = masker(vids[head][:m], boxes[p.stem])
            rgbm = np.concatenate([rgb[..., :3], masks[..., None]], axis=-1)
            qc_dir = dst.with_name(dst.stem + "_mask_qc")
            qc_dir.mkdir(exist_ok=True)
            save_mask_qc(qc_dir / f"{p.stem}.jpg", vids[head][:m], masks, boxes[p.stem])
            n_empty = int((masks.reshape(m, -1).max(axis=1) == 0).sum())
            if n_empty:
                empty_report.append((p.stem, n_empty, m))
            print(f"    SAM 分数 {score:.3f}  首帧掩码占比 {masks[0].mean():.3f}  "
                  f"掩码为空 {n_empty}/{m} 帧")
        elif a.mask == "alpha" and rgb.shape[-1] == 4:
            rgbm = rgb
        else:
            mask = np.full(rgb.shape[:3] + (1,), 255, dtype=np.uint8)
            rgbm = np.concatenate([rgb[..., :3], mask], axis=-1)
        chunk["rgbm"] = rgbm

        for key, val in chunk.items():
            if key not in arrays:
                # 只有图像是逐帧随机读的，按 --chunk 切小块；state/action 训练开始时
                # 整个读进内存（streaming_replay_buffer.py），切 2 帧一块只会多出
                # 上万个小文件，经移动硬盘/Windows 中转时拷得很慢
                n_chunk = a.chunk if val.ndim > 2 else 4096
                arrays[key] = data.zeros(
                    key, shape=(0,) + val.shape[1:], dtype=val.dtype,
                    chunks=(n_chunk,) + val.shape[1:], compressor=comp)
            arrays[key].append(val)
        total += m
        ends.append(total)
        done.append(p.stem)
        tasks.add(str(d["task"]) if "task" in d.files else "")
        print(f"  {p.name} -> {m} 帧（累计 {total}）")

    if not ends:
        print("  一条都没转成")
        return 1
    meta.array("episode_ends", np.asarray(ends, dtype=np.int64),
               chunks=(len(ends),), compressor=None)
    # 来源信息放根组 attrs：loader 只遍历 meta/ 下的数组，attrs 不会被当成数据读进去
    root.attrs.update({
        "converter": "数据_硬件/灵巧手遥操/VR设备/to_dexgraspvla.py",
        "src": str(src),
        "episodes": done,
        "task": sorted(tasks),
        "fps": float(d0["fps"]) if "fps" in d0.files else None,
        "wrist_cam": wrist, "head_cam": head,
        "image_order": "BGR, HWC",
        "action": a.action,
        "arm_norm": "q_norm = clip((q - (hi+lo)/2) / ((hi-lo)/2), -1, 1)",
        "joint_min": jlo[:arm_dof].tolist(), "joint_max": jhi[:arm_dof].tolist(),
        "ee_names": list(ee_names), "ee_range": [float(ee_lo), float(ee_hi)],
        "ee_note": "0 = 合拢，1 = 全开",
        "mask": a.mask,
        "head_bbox": {k: boxes[k] for k in done} if masker is not None else None,
    })

    print(f"\n  写好了 {dst}")
    print(f"  {len(ends)} 条 / {total} 帧")
    for key in ("right_cam_img", "rgbm", "right_state", "action"):
        print(f"    data/{key:14s} {arrays[key].shape}  {arrays[key].dtype}")
    print(f"    meta/episode_ends  ({len(ends)},) int64")

    if masker is not None:
        print(f"\n  掩码质检图：{dst.with_name(dst.stem + '_mask_qc')}/")
        for stem, n_empty, m in empty_report:
            print(f"  ⚠ {stem}：{n_empty}/{m} 帧掩码为空（Cutie 跟丢或被手完全挡住），看质检图")
    if dim != 13:
        print(f"\n  ⚠ 维度是 {dim}，不是 DexGraspVLA 默认的 13（7 臂 + 6 因时手）。")
        print(f"    训练时覆盖 shape_meta，并从官方 ckpt 部分加载（形状不符的两端随机初始化）：")
        print(f"        'task.shape_meta.obs.right_state.shape=[{dim}]' "
              f"'task.shape_meta.action.shape=[{dim}]'")
        print(f"        \"policy.start_ckpt_path='<官方 ckpt>'\" +training.start_ckpt_skip_mismatch=true")
    if a.mask == "none":
        print(f"\n  ⚠ rgbm 的第 4 通道是全 255 的占位掩码。")
        print(f"    掩码是他们定位目标物体的唯一输入，**这样训不出对的东西**。")
        print(f"    要真用得先接 SAM 生成掩码、Cutie 逐帧跟踪。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
