import time
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from py_bullet_env import UR5BulletEnv
from kinematics import UR5_KINEMATICS
import Vision_detection as vd
import control as ctrl
import control2 as ctrl2
import control3 as ctrl3

# =========================== SET UP =====================================================

def setup_sim(gui=True):
    env = UR5BulletEnv(gui=gui)
    robot = env.robot_id
    joints = ctrl2.get_revolute_joints(robot)
    prismatic_joints = ctrl2.get_prismatic_joints(robot)
    ctrl2.disable_default_motors(robot, joints)
    return env, robot, joints,prismatic_joints, UR5_KINEMATICS()

# ============================= MAIN ================================================
    
if __name__ == "__main__":

    # ======================================================
    # SIMULATION SETUP
    # ======================================================
    print("======================= STARTING FROM DROP LOCATION ======================")
    
    env, robot, joints, prismatic_joints, kinematics = setup_sim(gui=True)
    t=0
    t1=0
    t2=0
    step=0
    dt = 1.0 / 420.0
    REACH = 0      # joint-space motion to catch pose
    TRACK = 1      # Cartesian tracking of cube
    GRASP = 2      # force-based grasp
    DONE  = 3

    # ======================================================

    piece= 1-1/3.5
    camPos1 = np.array([env.belt_position[0]+0.05,-piece*(env.belt_length/2), 1.0])
    camPos2 = np.array([env.belt_position[0]-0.05,-piece*(env.belt_length/2), 1.0])
    camTarget = np.array([camPos1[0]-0.05,camPos1[1],0])#self.belt_position

    # ======================================================
    # INITIAL CUBE DROP
    # ======================================================
    #cube_pos_init = env.cube_pose["position"]
    z_off       = 0.15
    limit1      = 150
    z_des_catch = env.cube_size + env.belt_position[2] - 0.020
    res1,t = ctrl2.drop_in_tray(env, robot, joints, z_des_catch, limit1,t)
    t_log1=np.arange(0,t,dt)
    
    # ======================================================
    # ROBOT INITIAL STATE
    # ======================================================
    print("======================= READING POSITIONS ======================")
    print("Catch and drop height of End Effector: ",z_des_catch)
    q0, dq0 = ctrl2.get_current_joint_states(robot, joints)

    q0_ = -kinematics.ik_to_urdf(-q0)
    T0  = kinematics.fk(q0_)
    ctrl3.draw_frame(T0)
    Pr0 = T0[:3, 3]
    
    # ==========================================================
    # GETTING CUBE LOCATION FROM CAMERA
    # ==========================================================
    
    for j in joints: p.setJointMotorControl2(robot,j,controlMode=p.POSITION_CONTROL,targetPosition=q0[j-1],force=1000)
    target, q_catch,cube_yaw,t_catch, Vc, X2,t = ctrl.cube_data(env, kinematics, q0, Pr0, z_off,t)

    pc = target[:3]
    T_catch = kinematics.T_from_XYZ_RPY(target)
    R_d = T_catch[:3, :3]
    w_d = np.zeros(3)

    # ======================================================
    # CURRENT EE STATE
    # ======================================================
    q, dq = ctrl2.get_current_joint_states(robot, joints)
    jaq   = -kinematics.ik_to_urdf(-q)
    T_ee  = kinematics.fk(jaq)
    p0    = T_ee[:3, 3]

    # ======================================================
    # LOGGING VARIABLES
    # ======================================================
    t_log2, p_des_log_l, p_act_log_l = [], [], []
    cube_log_l, torque_log_l = [], []
    q_log_l, dq_log_l, q_catch_log_l = [], [], []
    p_q_log_l, contact_f_l = [], []
    EE_yaw=[]

    dq_max  = 3.333
    tau_max = 800
    eint = np.zeros((6, 1))

    Kp_val = 500
    Kd_val = 3 * np.sqrt(Kp_val)
    Ki_val = Kp_val / 2

    Kp = np.eye(6) * Kp_val
    Ki = np.eye(6) * Ki_val
    Kd = np.eye(6) * Kd_val

    bb = True
    grip = False
    grasp_time = 0

    mg = 9.81 * env.cube_mass
    force_threshold = mg / (2 * 0.3)

    t_max = np.linalg.norm(-target[:2] + np.hstack([X2])) / np.linalg.norm(Vc)

    t_wait   = t_max / 4
    t_track  = t_max / 4
    t_safe   = t_max / 8
    t_grasp  = t_max - (t_wait + t_track + t_safe)

    # ======================================================
    # MAIN CONTROL LOOP
    # ======================================================

    ctrl2.disable_default_motors(robot,joints)
    
    print("============= IMPLEMENTING CUBIC TRAJECTORY IN JOINT SPACE & LINEAR TRAJECTORY IN TASK SPACE WRT TIME =============")
    while t1 <= t_catch + t_max:

        grasped = False

        # ----- Read robot state -----
        q, dq = ctrl2.get_current_joint_states(robot, joints)
        dq = np.clip(dq, -dq_max, dq_max)

        jaq  = -kinematics.ik_to_urdf(-q)
        T_ee = kinematics.fk(jaq)
        x,y,z,roll,pitch,ee_yaw=kinematics.pRPY(T_ee)
        p_act=np.hstack([x,y,z])

        # ==================================================
        #           PHASE 1: REACH CATCH POSE
        # ==================================================
        if t1 <= t_catch:

            q_cmd, dq_cmd, ddqd = ctrl.cubic_joint_trajectory(
                env, kinematics, t1, t_catch, q0_, dq0, q_catch, dq_max
            )

            qd_off = kinematics.ik_to_urdf(q_cmd)

            q_catch_log_l.append(q_catch.copy())
            p_des_log_l.append(p_act.copy())

        # ==================================================
        #           PHASE 2: CARTESIAN TRACKING 
        # ==================================================
        else:

            if abs(t1 - t_catch) < dt:
                eint[:] = 0
                p_d = p_act.copy()

            wait = (t1 - t_catch) >= t_wait

            p_d, dp_d, ddp_d = ctrl.lineTrajectory(wait, dt, t_track, z_des_catch, Vc, z_off, p_act, p_d, bb)

            rpy_d = np.array([180, 0, np.rad2deg(cube_yaw)])
            q_catch1, *_ = kinematics.ik_one(np.hstack([p_d, rpy_d]), q)

            q_cmd, dq_cmd, ddqd = ctrl.CLIK(env, kinematics, jaq, q_catch1, dp_d)

            qd_off = kinematics.ik_to_urdf(q_cmd)

            q_catch_log_l.append(q_catch1.copy())
            p_des_log_l.append(p_d.copy())
            
        
        # ==================================================
        # TORQUE CONTROL
        # ==================================================
        tau, e, ed, eint = ctrl2.spong_torques_control(Kp, Ki, Kd, eint, joints, q, dq, qd_off, dq_cmd, ddqd)

        tau = np.clip(tau.flatten(), -tau_max, tau_max)
        
        for i, j in enumerate(joints): p.setJointMotorControl2(robot, j, controlMode=p.TORQUE_CONTROL, force=float(tau[i]))
        
        # ==================================================
        # LOGGING
        # ==================================================
        t_log2.append(t)
        torque_log_l.append(tau)
        p_act_log_l.append(p_act.copy())
        q_log_l.append(q.copy())
        dq_log_l.append(dq.copy())
        cube_log_l.append(env.pos_new_cube)
        EE_yaw.append(ee_yaw.copy())

        # ----- Step simulation -----
        env.step()
        time.sleep(dt)
        t += dt
        t1+=dt

    # ======================================================
    # POST-GRASP DROP
    # ======================================================
    #if grasp_time >= t_grasp:
    #    z_off = 0.15
    #    limit1 = 200
    #    ctrl2.drop_in_tray(env, robot, joints, z_off, cube_pos_init, limit1)

    # ======================================================
    # FINAL PLOTS
    # ======================================================
    
    
    p_q_log     = np.squeeze(p_q_log_l)
    c_f_log     = np.squeeze(contact_f_l)
    p_des_log   = np.squeeze(p_des_log_l)
    p_act_log   = np.squeeze(p_act_log_l)
    q_log       = np.squeeze(q_log_l)
    dq_log      = np.squeeze(dq_log_l)
    cube_log    = np.squeeze(cube_log_l)
    torque_log  = np.squeeze(torque_log_l)
    q_catch_log = np.squeeze(q_catch_log_l)
    EE_yaw_log     = np.squeeze(EE_yaw)

    # ================= DROP ========================================

    print("=================== GOING TO DROP LOCATION ======================")

    #limit2      = 150
    #res2,t = ctrl2.drop_in_tray(env, robot, joints, z_des_catch, limit1,t)

    print("======================= COMPLETED TRACKING THE RANDOMLY PLACED AND ORIENTED CUBE WORK PIECE ======================")
    print("Total loop time: ",t)

    print("======================= PLOTS ========================")

    # PHASE 2
    ctrl3.plot_all(env,p_des_log, p_act_log,  q_log, torque_log, t_log1,t_log2, cube_log, q_catch_log,EE_yaw_log,res1, pc)

