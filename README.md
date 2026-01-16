<h1 align="center">UR5 Robotic Pick-and-Place with Conveyor Tracking</h1>

<p align="center">
<b>Technologies:</b> PyBullet · Python · Robotics Kinematics · Optimization-Based Control · Computer Vision
</p>

<h2>Overview</h2>
<p align="justify">
This project presents a physics-based simulation of a <b>6-DOF UR5 robotic manipulator</b> designed to detect, track,
and intercept <b>randomly oriented workpieces on a moving conveyor</b>. The system integrates vision-based localization,
analytic kinematics, and real-time closed-loop inverse kinematics (CLIK) to achieve smooth and stable pick-and-place
operation under dynamic conditions.
</p>



<h2>Key Features</h2>
<ul>
  <li>Real-time tracking of moving workpieces with arbitrary orientation</li>
  <li>Analytic Forward and Inverse Kinematics for UR5</li>
  <li>Cubic joint-space trajectory generation for smooth motion</li>
  <li>Closed-Loop Inverse Kinematics with Damped Least Squares (DLS)</li>
  <li>Reduced-order dynamic modeling for physically consistent torque control</li>
  <li>Full digital twin simulation using PyBullet</li>
</ul>



<h2>System Architecture</h2>

<h3>1. Vision-Based Workpiece Localization</h3>
<p align="justify">
Canny edge detection is used to estimate workpiece pose (position and yaw). Pose updates are continuously synchronized
with conveyor motion for real-time interception.
</p>

<h3>2. Kinematic Control</h3>
<p align="justify">
Forward Kinematics (FK) computes the real-time end-effector pose, while Analytic Inverse Kinematics (IK) generates joint
configurations to reach the predicted interception region.
</p>

<h3>3. Trajectory Planning (Joint Space)</h3>
<p align="justify">
Smooth joint motion is generated using cubic joint-space polynomials, ensuring continuity of position and velocity:
</p>

<p align="center">
q(t) = at<sup>3</sup> + bt<sup>2</sup> + ct + d
</p>

<p align="center">
d = q<sub>0</sub>, &nbsp; c = q̇<sub>0</sub><br>
a = [2(q<sub>0</sub> − q<sub>f</sub>) + (q̇<sub>0</sub> + q̇<sub>f</sub>)t_catch] / t_catch<sup>3</sup><br>
b = [3(q<sub>f</sub> − q<sub>0</sub>) − (2q̇<sub>0</sub> + q̇<sub>f</sub>)t_catch] / t_catch<sup>2</sup>
</p>

<h3>4. Reduced-Order Dynamic Approximation</h3>
<p align="justify">
To enable real-time torque application without full rigid-body dynamics:
</p>
<ul>
  <li>Base and arm joints modeled as a yaw-dominant 3-link pendulum</li>
  <li>Wrist joints approximated as a simple pendulum with roll axis</li>
</ul>
<p align="justify">
This captures dominant inertial effects while remaining computationally efficient.
</p>

<h3>5. Conveyor Synchronization via CLIK (DLS)</h3>
<p align="justify">
Real-time interception is achieved using Closed-Loop Inverse Kinematics with Damped Least Squares:
</p>

<p align="center">
H q̇ = g
</p>

<p align="center">
H = J<sup>T</sup>J + λI , &nbsp; λ = 10<sup>−6</sup><br>
g = J<sup>T</sup>ẋ<sub>d</sub>
</p>

<p align="center">
q̇ = clip(H<sup>−1</sup>g, −q̇<sub>max</sub>, q̇<sub>max</sub>)
</p>



<h2>Results</h2>
<ul>
  <li>Stable tracking of moving workpieces under varying conveyor speeds</li>
  <li>Smooth interception trajectories with bounded joint velocities</li>
  <li>Accurate task-space tracking with minimal pose error</li>
  <li>Physically consistent interaction with the simulated environment</li>
</ul>



<h2>Future Extensions</h2>
<ul>
  <li>Full rigid-body dynamics with computed-torque control</li>
  <li>Model Predictive Control (MPC) for interception timing</li>
  <li>Learning-based grasp planning</li>
  <li>Multi-camera perception pipeline</li>
</ul>



<h2>Visual Results</h2>

<p align="center">

<b>Figure 1: UR5 CAD Assembly (PTC Creo)<br></b><br><i>High-fidelity CAD model used for mass properties and center-of-gravity estimation.</i><br>
<img src="https://github.com/user-attachments/assets/0064310f-ba77-4c0c-8835-1862cf67750a" width="800"><br>
<br><br>

<b>Figure 2: Physics-Based Simulation Environment (PyBullet)<br></b><br><i>Digital twin including conveyor, tray system, and UR5 manipulator.</i>
<br>
<img src="https://github.com/user-attachments/assets/3faf17c4-9afb-4a6b-af81-cb04a94e0620" width="1000"><br>
<br>

<b>Figure 3: Vision-Based Workpiece Localization<br></b><br><i>Canny edge detection pipeline for estimating workpiece position and orientation.</i>
<br>
<img src="https://github.com/user-attachments/assets/0525d75f-e6a8-42ad-915b-7f41ce874ac5" width="700"><br>
<br>

<b>Figure 4: End-Effector Trajectory during Catch-Point Approach<br></b><br><i>Predicted interception trajectory synchronized with conveyor motion.</i>
<br>
<img src="https://github.com/user-attachments/assets/62af23bf-d1eb-4836-b2ec-4b0143c540ac" width="1000"><br>
<br>

<b>Figure 5: Control System Performance<br></b><br><i>Joint torques, joint angles, and tracking errors across all operational phases.</i>
<br>
<img src="https://github.com/user-attachments/assets/e10de7a7-748e-4931-afc4-60235f826d1f" width="1000"><br>
<br>

<b>Figure 6: Task-Space Tracking Correlation<br></b><br><i>Correlation between end-effector pose and actual moving workpiece configuration.</i>
<br>
<img src="https://github.com/user-attachments/assets/bf706bcb-3b8b-495e-bc05-8b0b0f653a7a" width="800"><br>
<br>

<b>Figure 7: Full Cycle Path Visualization<br></b><br><i>End-effector path from tray to randomly oriented moving workpieces.</i>


</p>
