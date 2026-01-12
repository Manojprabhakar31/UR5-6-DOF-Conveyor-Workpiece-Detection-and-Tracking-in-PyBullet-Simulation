import time
import numpy as np
import pybullet as p
from kinematics import UR5_KINEMATICS
from dynamics import compute_MCG_6dof

kinematics=UR5_KINEMATICS()

#============================ HELPER FUNCTIONS ============================================

def get_revolute_joints(robot):
    """Return revolute joint indices in URDF order"""
    return [
        i for i in range(p.getNumJoints(robot))
        if p.getJointInfo(robot, i)[2] == p.JOINT_REVOLUTE
    ]

def get_prismatic_joints(robot):
    """Return prismatic joint indices in URDF order"""
    return [
        i for i in range(p.getNumJoints(robot))
        if p.getJointInfo(robot, i)[2] == p.JOINT_PRISMATIC
    ]

def get_current_joint_states(robot, joint_indices):
    """Read current joint states from PyBullet (rad) and (rad/sec) """
    q=  np.array([p.getJointState(robot, j)[0] for j in joint_indices])
    dq= np.array([p.getJointState(robot, j)[1] for j in joint_indices])
    return q, dq

def disable_default_motors(robot, joint_indices):
    for j in joint_indices:
        p.setJointMotorControl2(
            bodyUniqueId=robot,
            jointIndex=j,
            controlMode=p.VELOCITY_CONTROL,
            force=0
        )

# ==========================================================================================

def task_space_error(q, T_des):

    T_cur = kinematics.fk(q)

    config=kinematics.pRPY(T_cur)
    des_config=kinematics.pRPY(T_des)

    pos=config[0:3]
    ori = config[3:6]
    pos_des=des_config[0:3]
    ori_des = des_config[3:6]

    e_pos=pos-pos_des
    e_ori=ori-ori_des

    return np.linalg.norm(e_pos), np.linalg.norm(e_ori), pos, np.rad2deg(ori)
    
#============== INVERSE KINEMATICS ==================================
def solve_kinematics(kinematics, T_target, q_current):
    
    qs_all   = kinematics.ik_all(T_target)
    qs_valid = kinematics.filter_joint_limits(qs_all)

    if len(qs_valid) == 0:
        return None

    q_best,ee_pos, ee_rpy, pos_err_best, rot_err_best = kinematics.best_solution_fk_priority(
        qs_valid,
        T_target,
        q_current=q_current
    )
    #print("position from FK: ",ee_pos)
    #print("orientation from FK: ",ee_rpy)
    return q_best

#================ CONTROL OF DYMANICS =============================================

def position_control(
    robot,
    joint_indices,
    q_target,
    force=300.0
):
    """Pure position control (used for motion / IK)"""

    for joint_id, q in zip(joint_indices, q_target):
        p.setJointMotorControl2(
            bodyUniqueId=robot,
            jointIndex=joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=float(q),
            force=force
        )

#--------------------LINEARISED PD CONTROL-------------------------------

def spong_torques_control(Kp,Ki, Kd, e_int, arm_joint_indices, q, dq, q_des, dq_des, ddq_des):
    
    # ---------- Setup ----------

    arm_dof = len(arm_joint_indices)
    max_force = 800.0

    dyn_params = {
        "g": 9.81,
        "m1": 11.861916, "m2": 27.490578, "m3": 19.185826,
        "m4_1": 22.3843771, "m4_2": 3.1985511,
        "m5": 3.2763593, "m6": 1.1425590, "mp": 0.70443738,
        "l1": 0.425, "l2": 0.39225, "l3": 0.09475,
        "l4": 0.047375, "l5": 0.04625, "l6": 0.07725,
        "r1": 0.116, "r2": 0.090, "r3": 0.075,
        "r4": 0.075, "r5": 0.075, "r6": 0.075
    }

    # ---------- Current joint states (column vectors) ----------
    
    q=q.reshape(-1, 1)
    dq=dq.reshape(-1, 1)

    # ---------- Desired trajectories (column vectors) ----------

    q_des   = np.asarray(q_des[:arm_dof]).reshape(-1, 1)
    dq_des  = np.asarray(dq_des[:arm_dof]).reshape(-1, 1)
    ddq_des = np.asarray(ddq_des[:arm_dof]).reshape(-1, 1)
 
    # ----------EOM of System (DYNAMICS) ----------
    #M, C, G = compute_MCG_6dof(q, dq)
    M, C, G = compute_MCG_6dof(q, dq, dyn_params)
    M_arm = M[:arm_dof, :arm_dof]
    C_arm = C[:arm_dof, :arm_dof]
    G_arm = G[:arm_dof].reshape(-1, 1)

    # ---------- Linearised PD CONTROL ------------
    dt=1/420
    e  = q_des - q
    ed = dq_des - dq
    
    e_int =e_int + e * dt
    
    ddq_cmd = ddq_des + Kd @ (ed) + Kp @ e + Ki @ e_int
    tau = M_arm @ ddq_cmd + C_arm @ dq + G_arm   # (n×1)

    # ---------- RETURN TORQUE (PyBullet compatible) ----------

    tau = tau.flatten()
    tau = np.clip(tau, -max_force, max_force)

    return tau,e,ed,e_int

# ======================= Main Pipeline function ==============================

def move_ur5_to_pose(Kp,Ki, Kd, env,robot, joint_indices, kinematics, xyz_rpy,dq_des,ddq_des, limit,t):

    # ------------ Setup ---------------

    
    dt = 1 / 420.0
    torque       = np.zeros((6, limit))
    joint_angles = np.zeros((6, limit))
    error        = np.zeros((6, limit))
    errord       = np.zeros((6, limit))

    # ---------- IK ----------

    T_target  = kinematics.T_from_XYZ_RPY(xyz_rpy)
    q_current, dq = get_current_joint_states(robot, joint_indices)

    q_best = solve_kinematics(kinematics, T_target, q_current)
    if q_best is None:
        raise RuntimeError("❌ No valid IK solution")
    
    
    # ---------- Desired state ---------

    q_des   = kinematics.ik_to_urdf(q_best)

    # ---------- CONTROL LOOP ----------
    eint=np.zeros([6,1])
    
    for step in range(limit):

        # ------------- State of System -------------------------
        
        q, dq  = get_current_joint_states(robot, joint_indices)

        # ------------- Compute Torque From EOM of UR5 ------------------
        
        tau, e, ed, eint = spong_torques_control(Kp,Ki,Kd,eint,joint_indices, q, dq, q_des, dq_des, ddq_des)

        # ------------- Applying Torque --------------------------

        for i, j in enumerate(joint_indices): p.setJointMotorControl2(robot, j, controlMode=p.TORQUE_CONTROL, force=float(tau[i]))
        p.stepSimulation()
        env.step() 
        time.sleep(dt)
        t+=dt
        
        # -------------------- LOGGING ---------------------

        torque[:, step]       = tau
        joint_angles[:, step] = q

        # --------------- Errors --------------------------

        error[:, step]        = e.flatten()
        errord[:, step]       = ed.flatten()
        e_pos, e_ori, pos, ori = task_space_error(q, T_target)

        if np.linalg.norm(e) < 1e-4 and np.linalg.norm(ed) < 1e-3:
            print("✅ Target reached")
            break

    #return q_best, torque, joint_angles, error, errord
    return {
        "q_best": q_best,
        "torque": torque,
        "joints": joint_angles,
        "error": error,
        "errord": errord
    },t

# ======================= DROP IN TRAY ==============================

def drop_in_tray(env,robot,joints,z_des_catch,limit1,t):

    pos=env.tray_position
    pos2=np.hstack([pos[0:2],0])+np.array([0,0,z_des_catch])

    target_home2 = np.hstack([pos2,180,0,90])#np.array([0.0, -0.0135, 1.0012, -90, 0, 180])#-0.912, 0.0135, 0.0892, 0, -90, -90
    dqd1   = np.zeros(6)
    ddqd1  = np.zeros(6)
    
    Kp_val = 200.0
    Kd_val = 3*((Kp_val)**0.5)
    Ki_val = Kd_val/2

    Kp = np.diag([1, 2, 2, 3, 1, 1]) * Kp_val    
    Ki = np.eye(6) * Ki_val
    Kd = np.eye(6) * Kd_val

    res1,t = move_ur5_to_pose(Kp,Ki,Kd,env, robot, joints, kinematics, target_home2,dqd1,ddqd1, limit1,t)

    return res1,t

# =======================================================================================
 
def move_prismatic_to_position():
    return


def compute_ee_target_from_cube(cube_pos, cube_yaw, z_offset):
    
    #print("Cube position:", cube_pos)
    #print("Cube yaw (deg):", np.rad2deg(cube_yaw))
    
    # --- Position ---
    ee_pos = np.array([0.0, 0.0, z_offset])

    # --- Orientation ---
    cube_rpy = np.array([0.0, 0.0, np.rad2deg(cube_yaw)])
    ee_rpy2 = np.array([180, 0.0, 0.0])

    # --- Build transform ---
    xyz_rpy=np.hstack([cube_pos,cube_rpy])
    xyz_rpy2=np.hstack([ee_pos,ee_rpy2])
    
    T_cube = kinematics.T_from_XYZ_RPY(xyz_rpy)
    T_cube_tool=kinematics.T_from_XYZ_RPY(xyz_rpy2)

    T_target = T_cube@T_cube_tool
    
    #---------- END EFFECTOR CONFIGURATION FOR CUBE LOCATION ---------------------
    
    t=kinematics.pRPY(T_target)
    p = t[0:3]
    r = np.rad2deg(t[3:6])
    target_config= np.hstack([p,r])
    
    return target_config , T_target, T_cube

def apply_joint_torques(robot, joints, tau, limit=800):
    tau = np.clip(tau.flatten(), -limit, limit)
    for i, j in enumerate(joints):
        p.setJointMotorControl2(
            robot, j,
            controlMode=p.TORQUE_CONTROL,
            force=float(tau[i])
        )

