Working code for MCBF-QP: A local-minimum-free reactive safe controller, combining the generalizability of regular Control Barrier Function Quadratic Programs (CBF-QPs) and the concave obstacle navigation ability of Modulation of Dynamical Systems (onM-Mod-DS) by Yifan Xue. Code Contributor: Yifan Xue, and Ze Zhang. 

## Idea
As prominent real-time safety-critical reactive control techniques, Control Barrier Function Quadratic Programs (CBF-QPs) work for control affine systems in general but result in local minima in the generated trajectories and consequently cannot ensure convergence to the goals. Contrarily, Modulation of Dynamical Systems (Mod-DSs), including normal, reference, and on-manifold Mod-DS, achieve obstacle avoidance with few and even no local minima but have trouble optimally minimizing the difference between the constrained and the unconstrained controller outputs, and its applications are limited to fully-actuated systems. We dive into the theoretical foundations of CBF-QP and Mod-DS, proving that despite their distinct origins, normal Mod-DS is a special case of CBF-QP, and reference Mod-DS's solutions are mathematically connected to that of the CBF-QP through one equation. Building on top of the unveiled theoretical connections between CBF-QP and Mod-DS, reference Mod-based CBF-QP and on-manifold Mod-based CBF-QP controllers are proposed to combine the strength of CBF-QP and Mod-DS approaches and realize local-minimum-free reactive obstacle avoidance for control affine systems in general. We validate our methods in both simulated hospital environments and real-world experiments using Ridgeback for fully-actuated systems and Fetch robots for underactuated systems. Mod-based CBF-QPs outperform CBF-QPs as well as the optimally constrained-enforcing Mod-DS approaches we proposed in all experiments. 
 
### How to use
The current version of MCBF-QP controllers are developed for robot navigation tasks and supports 2 robot models: differential drive (unicycle model) and omni-directional drive. Controller and environment settings can be updated in `env_controller.yaml` inside the `config` folder.

Test GPDF-based distance field generator.
```
cd src/
python3.8 test_gpdf_image.py 
```

Run the real-time visualization and debugger for MCBF-QP controllers.
```
cd src/
python3.8 ctrl_omcbf_debugger.py 
```

Generate gifs to record the testing results of multiple robots in a single environment using MCBF-QP controllers.
```
cd src/
python3.8 test_ctrl_omcbf_gif.py
```

