# DEWO v9.1

**主实验：scratch**（`train_dexjoco.py --init scratch`）。s0 冻 MoT，后做、不作为是否成立的判据。

\(V(s)=\mathbb{P}(\mathrm{S0\ success}\mid s)=k_t/10\)。不是折扣 return。没有 Pass@10 就不打 \(V\) 损失。

不改栈（224 / z-score，replan=24，NFE=10，评测 4×50）。不重扫、不新 collect。mix：\(\varepsilon_{\mathrm{cfg}}=\varepsilon_0+w(\varepsilon_+-\varepsilon_0)\)。

入口：`eval_dexjoco.py` / `collect_dexjoco.py` / `prepare_dexjoco.py` / `train_dexjoco.py`。
默认 `prepare --dewo-version v9.1`，`train --init scratch --gpus 0,1,2,3`。

---

## 池（四类）

Collect：seeds 10086–10135×4。Scan：失败从 48 起每 24 帧 Pass@10，第一个 0/10 = cliff \(M\)。

10 条续行：数 \(k_t\)。事件上第一条有 RGB 的成功 \(\tau^{(t)}\) 再切 33 帧当 D+。不要 stitch，不要 timeout 长尾。

| 类 | 内容 | action | video | \(V\) | 文本 |
|----|------|--------|-------|-------|------|
| D0 | **expert 全集** ∪ collect `one_per_all_success_seed`（RNG 20260820） | 开 | 开 | 关 | \(P\) |
| D_scan | 原始失败每个扫描格 33 帧，窗起点 \(=t\) | 关 | 关 | \(k_t/10\) | \(P\) |
| D_fail | 即 \(t=M\) 那条 D_scan | 关 | 开 | \(0/10\) | \(P\) Failed |
| D+(t) | 仅事件：\(\tau^{(t)}[t,t+33)\) | 开 | 开 | 关 | 90% Successful / 10% \(P\) |

事件 iff \(k_{t-24}-k_t\ge 3\) 且 \(k_t\ge 1\)。\(t=48\)、→0/10、噪声 1–2/10：无 D+，仍有 D_scan。

hammer collect 过长成功不进 D0。expert 不过滤 `expert_max`。leftover collect 成功 → val。

scratch：`dewo_scratch_pool`，不 pin，无 identity/action lock。`CFG_DROPOUT=0`，fast=0。Val 全 base。

---

## Value / 门

只在 D_scan / D_fail 窗起点约束。禁止插值、禁止 D0=1。Huber，tokens stop-grad，base 文本。

\(R_t=V_{t-1}-V_t\)。CFG ON iff \(R_t>\delta\) 且 \(V_t>\varepsilon_{\mathrm{alive}}\)。\(\delta=Q_{0.95}\) 用 **\(V_\phi\)** 在 \(k=10\) 前缀上的健康残差，不是标签差（两端 \(k=10\) 的 \(\Delta k\equiv 0\)）。禁止在 seeds 0–49 上扫阈值。先报 \(w=0\)。

---

## 训练

```bash
python scripts/prepare_dexjoco.py --task-name fold_glasses \
  --collect-dir collect_results/dexjoco/fold_glasses/<stamp> --gpus 0,1,2,3

python scripts/train_dexjoco.py --task-name fold_glasses --init scratch \
  --prepare-dir prepare_results/dexjoco/fold_glasses/<stamp> --gpus 0,1,2,3
```

\(w=0\) 不低于 S0 再考虑加 \(w\)。拟合不住 \(k/10\)、健康 \(R\) 与下降点不可分、D+ 残差不大于 D0：停。
