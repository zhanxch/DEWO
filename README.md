# DEWO: Direct Experience World-Model Optimization

<p align="center">
  <strong>Learning the World Beyond Action Imitation</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.37398"><img src="https://img.shields.io/badge/arXiv-2609.37398-b31b1b.svg" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/code-coming%20soon-lightgrey.svg" alt="Code coming soon">
  <img src="https://img.shields.io/badge/project%20page-coming%20soon-lightgrey.svg" alt="Project page coming soon">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.37398">Paper</a>
  &nbsp;·&nbsp;
  <a href="https://arxiv.org/pdf/2609.37398">PDF</a>
  &nbsp;·&nbsp;
  Code coming soon
  &nbsp;·&nbsp;
  Project page coming soon
</p>

<p align="center">
  Xiangcheng Zhan<sup>1</sup> &nbsp;·&nbsp;
  Zirui Chen<sup>2</sup> &nbsp;·&nbsp;
  Yicheng Zhao<sup>3</sup> &nbsp;·&nbsp;
  Ziteng Gao<sup>1</sup> &nbsp;·&nbsp;
  Shuo Yang<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Harbin Institute of Technology
  &nbsp;&nbsp;
  <sup>2</sup>Dalian University of Technology
  &nbsp;&nbsp;
  <sup>3</sup>Southern University of Science and Technology
</p>

---

> **Code, checkpoints, and the project page are coming soon.**
>
> This repository will host the official implementation of DEWO. The paper is available now on [arXiv](https://arxiv.org/abs/2609.37398).

## Abstract

World-Action Models (WAMs) couple action generation with predictions of how physical interactions unfold. However, current post-deployment learning paradigms typically improve behavior without requiring better world predictions. Especially in dexterous manipulation, small execution errors can compound in high-dimensional action spaces, hindering policy improvement and pushing interactions beyond the world model’s training distribution. Motivated by this, we propose Direct Experience World-Model Optimization (DEWO), a post-deployment learning paradigm for WAMs that, alongside action imitation, refines world representations through visual experience to better condition action generation. Specifically, it identifies interaction turning points and learns from successful and failed futures to support classifier-free guidance. An additional value head estimates task progress from video representations and activates guidance when progress stalls during inference. Across five DexJoCo tasks, DEWO improves average success across all three WAM formulations. Ablations show that visual supervision from successful and failed continuations improves both prediction and control beyond action supervision alone. On four real-world tasks across Wuji and Sharpa, 3×3 grid evaluations show that two rounds of deployment learning increase success from 51.0% to 71.7% in cells with at least one initial success, a gain of 20.7 percentage points. These findings support continued predictive learning for improving control through deployment experience, making world modeling an active part of WAM adaptation.

## 说明

论文已经公开。官方代码、模型权重和项目页即将发布，目前 `main` 只保留这一页。

尚未定稿的仓库快照保存在 [`draft_unfinal`](https://github.com/zhanxch/DEWO/tree/draft_unfinal) 分支，不作为正式发布。

## Citation

```bibtex
@misc{zhan2026directexperienceworldmodeloptimization,
  title={Direct Experience World-Model Optimization: Learning the World Beyond Action Imitation},
  author={Xiangcheng Zhan and Zirui Chen and Yicheng Zhao and Ziteng Gao and Shuo Yang},
  year={2026},
  eprint={2609.37398},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2609.37398}
}
```
