# Eikonal Rendering in Python

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

### Overview

This project implements a modern Python-based Eikonal rendering pipeline using PyTorch and Taichi for real-time rendering of complex optical effects like refraction, reflection, and anisotropic scattering in non-homogeneous media (e.g., glass, crystals).

### Key Features

-   **Real-time Rendering:** Achieves interactive frame rates for complex optical phenomena.
-   **Modern Python Stack:** Built with PyTorch (for tensor operations/auto-diff) and Taichi (for high-performance GPU kernels), making it accessible and integrable with ML tools.
-   **Comparative Study:** Implements and compares two wavefront propagation methods:
    -   **Point-based (Monte Carlo):** Simulates wavefronts using discrete points/photons.
    -   **Patch-based (Adaptive):** Simulates wavefronts using surface patches/slices.
-   **Efficient Storage:** Explores hierarchical data structures (Octree, MLP, SIREN) for compactly storing pre-computed irradiance data, reducing memory overhead.
-   **Volumetric Rendering:** Handles effects like scattering, absorption, and emission within translucent materials.

### Project Aims

1.  **Modernize the Codebase:** Re-implement Eikonal rendering in modern Python to overcome limitations of outdated C++/CUDA codebases (hard maintenance, lack of ML integration).
2.  **Compare Propagation Methods:** Systematically analyze point-based vs. patch-based wavefront propagation for accuracy and performance.
3.  **Explore Hierarchical Storage:** Evaluate compressed representations (Octree, Neural Fields) for storing volumetric irradiance data efficiently.

### Installation & Usage

See *requirements.txt*

---

<a name="chinese"></a>
## 中文

### 概述

本项目使用 PyTorch 和 Taichi 实现了一个基于现代 Python 的 Eikonal 渲染管线，用于实时渲染非均匀介质（如玻璃、晶体）中复杂的折射、反射和各向异性散射等光学效果。

### 主要特性

-   **实时渲染:** 针对复杂光学现象实现交互式帧率渲染。
-   **现代 Python 技术栈:** 基于 PyTorch（用于张量运算和自动微分）和 Taichi（用于高性能 GPU 计算）构建，易于使用并可集成机器学习工具。
-   **对比研究:** 实现并比较了两种波前传播方法：
    -   **基于点的方法 (蒙特卡洛):** 使用离散点/光子模拟波前。
    -   **基于面片的方法 (自适应):** 使用表面面片/切片模拟波前。
-   **高效存储:** 探索使用层次化数据结构（八叉树、MLP、SIREN）来紧凑存储预计算的辐照度数据，降低内存开销。
-   **体渲染:** 处理半透明材料内部的散射、吸收和自发光等效。

### 项目目标

1.  **代码库现代化:** 用现代 Python 重新实现 Eikonal 渲染，克服过时的 C++/CUDA 代码库的局限性（难以维护、缺乏 ML 集成）。
2.  **比较传播方法:** 系统分析基于点与基于面片的波前传播方法的准确性和性能。
3.  **探索层次化存储:** 评估压缩表示（八叉树、神经场）以高效存储体辐照度数据。

### 安装与使用

见*requirements.txt*

