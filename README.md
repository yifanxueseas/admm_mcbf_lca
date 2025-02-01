Working code for ADMM-MCBF-LCA: A Layered Control Architecture for Safe Real-Time Navigation by Anusha Srikantha and Yifan Xue. Code Contributor: Anusha Srikantha, Yifan Xue, and Ze Zhang. 

## Idea
 We consider the problem of safe real-time navigation of a robot in an unknown dynamic environment with moving obstacles and input saturation constraints. We assume that the robot detects nearby obstacle boundaries with a short-range sensor. This problem presents three main challenges: i) input constraints, ii) safety, and iii) real-time computation. To tackle all three challenges, we present a layered control architecture (LCA) consisting of an offline path library generation layer, and an online path selection and safety layer. To overcome the limitations of reactive methods, our offline path library consists of feasible controllers, feedback gains, and reference trajectories. To handle computational burden and safety, we solve online path selection and generate safe inputs that run at $100$ Hz. Through simulations on Gazebo and Fetch hardware in an unknown indoor environment, we evaluate our approach against baselines that are layered, end-to-end, or reactive. Our experiments demonstrate that among all algorithms, only our proposed LCA is able to complete tasks such as reaching a goal, safely. When comparing metrics such as safety, input error, and success rate, we show that our approach generates safe and feasible inputs throughout the robot execution.
 
### How to use
The current version of ADMM_MCBF_LCA framework is setup for robot navigation tasks using differential drive (unicycle model). Robot and environment settings can be updated in `env_controller.yaml` inside the `config` folder. Offline multi-path settings can be altered in `path_generator.py`.

Test GPDF-based distance field generator.
```
cd src/
python3.8 test_gpdf_image.py 
```

Run the real-time visualization and debugger for ADMM_MCBF_LCA.
```
cd src/
python3.8 test_admm_lca.py 
```

