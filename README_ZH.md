# MiMoCo

**MiMoCo: A Multimodal Imitation Learning Framework for Whole-Body Mobile Control with Exoskeleton-VR Teleoperation**

[中文](README_ZH.md) | [English](README.md)

[![IEEE IoT Journal](https://img.shields.io/badge/Journal-IEEE%20Internet%20of%20Things-blue)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

本仓库为论文 **MiMoCo**（IEEE Internet of Things Journal）的**官方开源项目**，提供从**外骨骼–VR 遥操作数据采集**到**全身移动操作模仿学习**的完整技术栈。我们将按论文结构持续发布代码、模型与文档；欢迎 Star 关注更新。

<!-- ## 📋 目录

- [🏠 简介](#-简介)
- [🧭 系统架构](#-系统架构)
- [🤖 外骨骼–VR 遥操作](#-外骨骼vr-遥操作)
- [🧠 MiMoCo 网络结构](#-mimoco-网络结构)
- [🧪 实验任务](#-实验任务)
- [🔥 开源进度](#-开源进度)
- [📚 快速开始](#-快速开始)
- [📁 仓库结构](#-仓库结构)
- [🔗 引用](#-引用)
- [📄 许可证](#-许可证)
- [👏 致谢](#-致谢) -->

---

<p align="center">
  <img src="figures/fig1_overview.png" alt="MiMoCo 系统总览" width="96%"/>
</p>
<p align="center"><em>图 1：MiMoCo 总览。外骨骼–VR 遥操作系统支持单人全身移动操控与简单力反馈，采集多模态示教；学习得到的模仿策略输出全身移动操作动作序列。</em></p>

---

## 🏠 简介

在工业物联网（IIoT）场景中，**全身移动操作**（双臂 + 移动底盘协同）对智能制造与柔性自动化至关重要，依赖高质量示教数据与长时域、多模态动作预测。MiMoCo 提出：

1. **外骨骼–VR 一体化遥操作**：单操作者完成全身移动操控，外骨骼提供关节级映射与简单力反馈，VR 提供沉浸 FPV 与底盘控制。  
2. **MiMoCo 模仿学习框架**（CVAE + 动作分块）  
   - **ECM-Net**（编码器）：线性复杂度的阶段级时序建模，缓解长程误差累积。  
   - **MRF-Net**（解码器）：全局感受野（GRF）+ 局部感受野（LRF）双路径视觉–运动融合，对齐宏观导航与微观手眼操作。

在真实轮式双臂机器人 **Robint** 上的四类任务（收纳、搬运、分拣、障碍清除）中，MiMoCo 在多个子任务上优于 ACT、Diffusion Policy、BeT 等基线。

---

## 🧭 系统架构

<p align="center">
  <img src="figures/fig2_system-architecture.png" alt="端到端系统架构" width="70%"/>
</p>
<p align="center"><em>图 2：端到端架构。左：外骨骼–VR 遥操作采集臂、夹爪与底盘指令；右：训练后的 MiMoCo 根据观测生成全身控制动作。</em></p>

| 阶段 | 内容 |
|------|------|
| **数据采集** | 三视角 RGB（头部、左右腕）+ 本体感知 + 18 维全身动作，50 Hz（$\Delta t = 20$ ms） |
| **策略学习** | ECM-Net 编码动作块潜变量 $\mathbf{z}$；MRF-Net 融合多相机视觉与 $\mathbf{s}_t$ 解码未来 $K$ 步动作 |
| **部署执行** | 推理时 $\mathbf{z}=\mathbf{0}$；重叠 chunk 指数加权时序融合（$m=0.01$）保证平滑 |

---

## 🤖 外骨骼–VR 遥操作

<p align="center">
  <img src="figures/fig3_teleop-system.png" alt="外骨骼VR遥操作与Robint平台" width="70%"/>
</p>
<p align="center"><em>图 3：外骨骼–VR 遥操作系统（左）与轮式移动机器人 Robint（右）。</em></p>

- **外骨骼双臂**：每臂 7 DoF，与人体手臂同构，实时关节角映射与过载力反馈。  
- **VR（Meta Quest 3）**：摇杆控制线/角速度，Trigger 控制夹爪，Grip 作为使能开关；头戴 FPV。  
- **主控与通信**：OrangePi Zero 2 + ROS 分布式节点；多源时间戳对齐与低通滤波。  
- **力反馈**：当 $\Delta\Theta(t) > \tau_{\mathrm{total}}$（部署中 $\tau_{\mathrm{total}}=0.15$ rad）时外骨骼锁定，提示接触过载。

---

## 🧠 MiMoCo 网络结构

<p align="center">
  <img src="figures/fig4_mimoco-architecture.png" alt="ECM-Net与MRF-Net结构" width="96%"/>
</p>
<p align="center"><em>图 4：左 ECM-Net—阶段级时序上下文；右 MRF-Net—GRF 宏观规划 + LRF 精细操作的双路径注意力。</em></p>

**输入**：多相机 RGB $\{i_t\}$，本体 $\mathbf{s}_t = [q_t, g_t, b_t] \in \mathbb{R}^{18}$（14 维双臂 + 2 维夹爪 + 2 维底盘速度）。  
**输出**：未来 $K=100$ 步动作块 $\hat{a}_{t:t+K}$（18 维全身目标）。  
**训练目标**：动作 L1 + 潜变量 KL（权重 10）；**推理**：$\mathbf{z}=\mathbf{0}$。

| 模块 | 论文设计 | 代码路径（持续更新） |
|------|----------|----------------------|
| ECM-Net | 可学习 $w_a$ 压缩全局 phase intent，门控检索历史 $K$ | `mimoco_models/models/` 编码器 |
| MRF-Net | LRF 局部窗口 $k{=}6$，GRF 全局 GAP，头比例 $\alpha{=}0.4$ | `mimoco_models/models/transformer.py` |
| 视觉骨干 | 每相机独立 ResNet-18 + 2D 正弦位置编码 | `mimoco_models/models/backbone.py` |
| 策略封装 | CVAE + chunk 解码 | `policy.py` → `ChunkSeqPolicy`（`CHUNK_SEQ`） |

**默认超参**（与论文 Sec. IV 一致）：$d_{\mathrm{model}}{=}512$，FFN $3200$，8 头，编码器 4 层 / 解码器 7 层，约 103.77M 参数。

---

## 🧪 实验任务

<p align="center">
  <img src="figures/fig5_task-box-stowing.png" width="48%" alt="Object Stowing"/>
  <img src="figures/fig6_task-box-transport.png" width="48%" alt="Object Transport"/>
</p>
<p align="center">
  <img src="figures/fig7_task-object-sorting.png" width="48%" alt="Object Sorting"/>
  <img src="figures/fig8_task-obstacle-clearance.png" width="48%" alt="Obstacle Clearance"/>
</p>
<p align="center"><em>图 5–8：Object Stowing、Object Transport、Object Sorting、Obstacle Clearance 任务设置。</em></p>

| 任务 | 子任务 | 要点 |
|------|--------|------|
| Object Stowing | Grasp → Place → Transfer | 双臂协作 + 长距离底盘移动（约 3.5 m） |
| Object Transport | Grasp → Move → Place | 通过窄门，约 7 m 行程 |
| Object Sorting | Identify → Move → Place | 双色箱体识别与分放 |
| Obstacle Clearance | Grasp → Clear → Move | 路障抓取、移开、通行 |

每任务 50 条示教，单操作者采集；图像 $480\times640$，三相机。

<p align="center">
  <img src="figures/fig10_exp.png" alt="四类任务真机执行" width="96%"/>
</p>
<p align="center"><em>图 10：MiMoCo 在四类全身移动操作任务上的顺序执行（头部宏观视角 + 腕部精细视角）。</em></p>

---

## 🔥 开源进度

✅ 已发布 · 🚧 即将推送

| 模块 | 状态 | 说明 |
|------|------|------|
| 数据管线（HDF5 加载、归一化、可视化） | ✅ | [`utils.py`](utils.py)、`EpisodicDataset`、[`visualize_episodes.py`](visualize_episodes.py) |
| 训练框架（`CHUNK_SEQ` + 基线） | ✅ | CVAE 动作分块 + Diffusion/CNNMLP，见 [`imitate_episodes.py`](imitate_episodes.py)、[`policy.py`](policy.py) |
| 实验基础设施（日志/权重/验证） | ✅ | W&B、`policy_best.ckpt`、`imitate_episodes.py --eval` |
| MiMoCo 核心模块（ECM-Net + MRF-Net） | 🚧 | Sec. III-B-2/III-B-3：阶段门控时序编码 + LRF/GRF 双路径注意力（$\alpha$、$k$） |
| 推理与建模对齐 | 🚧 | 时序融合（Alg. 2）、显式 18 维 $\mathbf{s}_t=[q,g,b]$、BeT 基线、ECM/MRF 消融 |
| 评估与分析脚本 | 🚧 | 真机 rollout/子任务指标 + 长程 MSE/注意力分析（Sec. IV） |

**遥操作 / 部署**

| 模块 | 状态 | 说明 |
|------|------|------|
| 资源与文档 | ✅ | 论文配图见 [`figures/`](figures/)（总览、架构、任务、实验） |
| 遥操作与部署栈 | 🚧 | 外骨骼–VR ROS 采集 + 力反馈 + 时间同步 + 真机执行节点 |
| 公共数据发布 | 🚧 | 示例数据包与格式说明见 [`data/README.md`](data/README.md) |

---

## 📚 快速开始

### 环境

1. Python 3.9+（依赖见 [`conda_env.yaml`](conda_env.yaml)）。  
2. 安装模型包：

```bash
cd mimoco_models && pip install -e . && cd ..
```

3. 使用 **Diffusion** 基线时需额外安装 `robomimic`、`diffusers` 等。

### 环境变量

| 变量 | 含义 |
|------|------|
| `MIMOCO_DATA_DIR` | 数据集根目录，默认 `data/` |
| `WANDB_PROJECT` | W&B 项目名（默认 `mimoco`） |
| `MIMOCO_PRETRAIN_CKPT` | `--load_pretrain` 时的预训练权重路径 |
| `MIMOCO_TRAIN_NUM_WORKERS` | DataLoader 进程数（默认 8） |

### HDF5 数据格式

每条 episode 一个 `.hdf5`，相机名与 `--camera_names` 一致：

| Key | 说明 |
|-----|------|
| `/observations/qpos` | 本体状态 |
| `/observations/images/<camera>` | RGB，如 `cam_high`, `cam_left_wrist`, `cam_right_wrist` |
| `/action` | 臂 + 夹爪目标 |
| `/base_action` | 底盘速度（可选；缺失时补零，动作维数为 18） |

### 训练 MiMoCo（`CHUNK_SEQ`）

```bash
export MIMOCO_DATA_DIR=./data/my_dataset

python imitate_episodes.py \
  --ckpt_dir ./checkpoints/run1 \
  --policy_class CHUNK_SEQ \
  --dataset_dir "$MIMOCO_DATA_DIR" \
  --episode_len 400 \
  --camera_names cam_high,cam_left_wrist,cam_right_wrist \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 8 \
  --dim_feedforward 3200 \
  --lr 1e-5 \
  --seed 0 \
  --num_steps 200000
```

```powershell
$env:MIMOCO_DATA_DIR = ".\data\my_dataset"
python imitate_episodes.py --ckpt_dir .\checkpoints\run1 --policy_class CHUNK_SEQ `
  --dataset_dir $env:MIMOCO_DATA_DIR --episode_len 400 `
  --camera_names cam_high,cam_left_wrist,cam_right_wrist `
  --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 `
  --dim_feedforward 3200 --lr 1e-5 --seed 0 --num_steps 200000
```

验证 checkpoint：

```bash
python imitate_episodes.py --eval --ckpt_dir ./checkpoints/run1
```

可视化 episode：

```bash
python visualize_episodes.py --dataset_dir "$MIMOCO_DATA_DIR" --episode_idx 0
```

控制频率与论文一致：[`constants.py`](constants.py) 中 `FPS=50`，`DT=0.02`。

---

## 📁 仓库结构

| 路径 | 说明 |
|------|------|
| [`figures/`](figures/) | 论文插图（PDF/PNG） |
| [`imitate_episodes.py`](imitate_episodes.py) | 训练 / 验证入口 |
| [`policy.py`](policy.py) | MiMoCo与基线 |
| [`mimoco_models/`](mimoco_models/) | ResNet 骨干、Transformer、SeqVAE |
| [`utils.py`](utils.py) | 数据加载 |
| [`constants.py`](constants.py) | 控制周期常量 |

---

## 🔗 引用

若使用本代码或 MiMoCo 方法，请引用：

```bibtex
@article{mei2025mimoco,
  title={MiMoCo: A Multimodal Imitation Learning Framework for Whole-Body Mobile Control with Exoskeleton-VR Teleoperation},
  author={Mei, Jie and Wu, Xinkai and Zhang, Yue and Song, Tao and Xiong, Zhongxia},
  journal={IEEE Internet of Things Journal},
  year={2026}
}
```

## 📄 许可证

见 [`LICENSE`](LICENSE)。`mimoco_models` 遵循 Apache 2.0（[`mimoco_models/LICENSE`](mimoco_models/LICENSE)）。

## 👏 致谢

本仓库在系统搭建、数据采集与模仿学习实现方面受以下开源项目启发，并在部分工程组织与实现细节上参考了其代码：

- [ALOHA](https://github.com/tonyzhaozh/aloha): 遥操作与 HDF5 数据采集流程设计（多设备协同采集、数据记录与可视化脚本组织）为本仓库的数据采集与处理流程提供了重要参考。
- [ACT](https://github.com/tonyzhaozh/act): Action Chunking with Transformers 的训练范式、`imitate_episodes.py` 风格训练入口、CVAE + chunk 预测与评估流程启发了本仓库模仿学习训练管线。
- [OpenHomie](https://github.com/InternRobotics/OpenHomie): 面向全身移动操作与同构外骨骼遥操作的一体化系统思路，以及“硬件-策略-部署”模块化开源组织方式，对本项目的系统化开源设计有直接启发。
