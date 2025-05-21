# ADMM-MCBF-LCA: A Layered Control Architecture for Safe Real-Time Navigation  

## Overview  
This repository contains the implementation of **ADMM-MCBF-LCA**, a **Layered Control Architecture (LCA)** for real-time robot navigation in dynamic environments with arbitrary but smooth geometries. The approach was developed for the paper:  

**"ADMM-MCBF-LCA: A Layered Control Architecture for Safe Real-Time Navigation"**  
**Authors:** Anusha Srikanthan, Yifan Xue  
**Code Contributors:** Anusha Srikanthan, Yifan Xue, Ze Zhang  

## Idea  
We address the problem of **safe real-time navigation** for a robot operating in an **dynamic environment** with **moving obstacles** and **input saturation constraints**. The robot relies on short-range sensor data to detect nearby obstacle boundaries and also has access to obstacle motion.  

### **Key Challenges:**  
1. **Input Constraints** – Ensuring the robot respects actuator limitations.  
2. **Safety** – Guaranteeing collision avoidance while navigating.  
3. **Real-Time Computation** – Running at high frequencies (100 Hz) to maintain responsiveness.  

### **Proposed Solution:**  
To tackle these challenges, we introduce a **Layered Control Architecture (LCA)** consisting of:  
- **Offline Path Library Generation Layer** – Precomputes feasible controllers, feedback gains, and reference trajectories to avoid reactive method limitations.  
- **Online Path Selection & Safety Layer** – Selects an optimal path in real-time while enforcing safety constraints and maintaining computational efficiency.  

### **Key Results:**  
- Runs at **100 Hz**, ensuring real-time safety.  
- Outperforms **layered, end-to-end, and reactive** baselines in **safety, input feasibility, and task success rate**.  
- Successfully tested in **Gazebo simulations** and on **Fetch hardware** in an unknown indoor environment.  

## Installation & Dependencies  
This repository is implemented in Python 3.8 and requires the following dependencies:  

- python        3.8.10  
- numpy         1.22.4
- scipy         1.7.1
- matplotlib    3.7.5
- jax           0.4.13
- cvxpy         1.3.2
- opencv-python 3.4.8
- shapely       1.7.1
- pandas        1.4.3

## How to Use  

The **ADMM-MCBF-LCA** framework is currently set up for **robot navigation tasks** using a **differential drive (unicycle model)**.  

### **Configuration:**  
- Update **robot and environment settings** in `config/env_controller.yaml`.  
- Modify **offline multi-path settings** in `path_generator.py`.  

### **Run Tests & Visualizations:**  

#### **1. Test the GPDF-based distance field generator**  
```bash
cd src/
python3.8 test_gpdf_image.py
```

#### **2. Run real-time visualization and debugger for ADMM-LCA**  
```bash
cd src/
python3.8 test_admm_lca.py
```

## Citation
If you find our work useful, please cite our paper:

Srikanthan, A., Xue, Y., Kumar, V., Matni, N., & Figueroa, N. (2025). ADMM-MCBF-LCA: A Layered Control Architecture for Safe Real-Time Navigation. arXiv preprint arXiv:2503.02208.

Y. Xue and N. Figueroa, “No minima, no collisions: Combining modulation and control barrier function strategies for feasible dynamical collision avoidance,” 2025. Available: https://arxiv.org/abs/2502.14238

A. Srikanthan, V. Kumar, and N. Matni, “Augmented lagrangian methods as layered control architectures,” arXiv preprint arXiv:2311.06404, 2023.