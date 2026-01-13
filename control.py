import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import Vision_detection as vd
from kinematics import UR5_KINEMATICS
import control3 as ctrl3

kinematics=UR5_KINEMATICS()

# ========================= ENTRY AND EXIT POINTS IN WORK SPACE ========================================

def line_circle_intersection(x0, y0, psi, xr, yr, R, eps=1e-8):

    cos_p = np.cos(psi)
    sin_p = np.sin(psi)

    Pc0 = np.array([x0, y0])

    # ================= VERTICAL LINE (psi = ±90°) =================

    if abs(cos_p) < eps:
        # x = x0
        dx = x0 - xr
        disc = R**2 - dx**2
        if disc < 0:
            return None

        s = np.sqrt(disc)
        p1 = np.array([x0, yr + s])
        p2 = np.array([x0, yr - s])

    # ================= GENERAL LINE =================

    else:
        A = np.tan(psi)
        B = y0 - A * x0 - yr

        a = 1 + A**2
        b = 2 * (A * B - xr)
        c = xr**2 + B**2 - R**2

        disc = b**2 - 4 * a * c
        if disc < 0:
            return None

        s = np.sqrt(disc)
        x1 = (-b + s) / (2 * a)
        x2 = (-b - s) / (2 * a)

        p1 = np.array([x1, y0 + A * (x1 - x0)])
        p2 = np.array([x2, y0 + A * (x2 - x0)])

    # ================= ORDER BY DISTANCE FROM CURRENT CUBE POS =================

    if np.linalg.norm(Pc0 - p1) <= np.linalg.norm(Pc0 - p2): return np.vstack([p1, p2])
    else: return np.vstack([p2, p1])

# ======================= OPTIMAL CATCH POINT SEARCH ==============================================

def find_optimal_catch_point_jacobian(env,kinematics,X1, X2, Pc0,Oc0, Pr0, Vc, q_seed, dq_max,z_off,c=0.5, num_steps=100):

    d12 = X2 - X1
    L = np.linalg.norm(d12)
    u = d12 / L

    for d in np.linspace(0.125/2 * L, 0.5 * L, num_steps):

        Pc = X1 + d * u
        Pc3 = np.array([Pc[0], Pc[1], z_off])
        orn=np.array([180,0,np.rad2deg(Oc0)])
        t = np.linalg.norm(Pc3 - Pc0) / np.linalg.norm(Vc)
        if t <= 0: continue

        vr = (Pc3 - Pr0) / t

        q_sol,_,_,_,_ = kinematics.ik_one(np.hstack([Pc3,orn]), q_seed)
        if q_sol is None: continue

        Jv = kinematics.jacobian(q_sol)[:3, :]
        dq = np.linalg.pinv(Jv, rcond=1e-4) @ vr

        if np.all(np.abs(dq) <= c * dq_max):
            T = kinematics.fk(q_sol)
            rpy = np.array([180,0,np.rad2deg(Oc0)])#np.array([180,0,90])#np.rad2deg(kinematics.pRPY(T)[3:6])
            return np.hstack([Pc3, rpy])

    return None


# ==================== DATA ABOUT CUBE MAIN PIPELINE ====================================

def make_camera(camPos,camTarget):
    camz=camTarget-camPos
    camz=camz/np.linalg.norm(camz)
    camy=np.array([0,-1,0])
    camx=np.cross(camy,camz)
    camx=camx/np.linalg.norm(camx)
    Q = np.vstack((camx, camy, camz))   # columns are camera axes in world coordinates
    Q= np.transpose(Q)
    camPos=np.asarray(camPos).reshape(3, 1)
    T=np.vstack([np.hstack([Q,camPos]),np.array([0,0,0,1])])
    return T

def cube_vision_data(env,t):

    piece= 1-1/6
    camPos1 = np.array([env.belt_position[0]+0.05,-piece*(env.belt_length/2), 0.50])
    camPos2 = np.array([env.belt_position[0]-0.05,-piece*(env.belt_length/2), 0.50])
    camTarget = np.array([camPos1[0]-0.05,camPos1[1],0])#self.belt_position
    dt=1/420
    cal_3D=[]
    act=[]
    error=[]
    yaw_cube=[]
    error_yaw=[]
    step=0
    step1=0
    
    while True:
        
        corners=[]

        env.step()
        time.sleep(dt)

        if step%25==0:

            t1=0
            img,img_width,img_height,camFOV,_ = vd.image_feed(camPos1, camTarget)
            edges, f_edges_p1,XY_new1, XY_p1, upper_edges1,polar1,polar_smt1,polar_all1,points1,hull1,points1_p,hull1_p,best_name1 = vd.process_image(img,img_width,0)
            
            img2,img_width,imgHeight,camFOV,_=vd.image_feed(camPos2, camTarget)
            edges2, f_edges_p2,XY_new2,XY_p2,upper_edges2,polar2,polar_smt2, polar_all2,points2,hull2,points2_p,hull2_p,best_name2 = vd.process_image(img2,img_width,1)
            
            T1=make_camera(camPos1,camTarget)
            T2=make_camera(camPos2,camTarget)

            XY_p11,XY_p21,XY_new11,XY_new21=np.array([XY_p1]),np.array([XY_p2]),np.array([XY_new1]),np.array([XY_new2])
            XY_cam1=np.transpose(XY_p11/2+XY_new11/2)
            XY_cam2=np.transpose(XY_p21/2+XY_new21/2)

            point_3D=vd.stereo_triangulate_edges_centered(img_width, img_height, camFOV, T1, T2, XY_cam1, XY_cam2)
            corners_cam1,_,_,_,_,_=points1
            corners_cam2,_,_,_,_,_=points2
            corners_cam1=np.squeeze(np.array([corners_cam1])).T
            corners_cam2=np.squeeze(np.array([corners_cam2])).T
            
            for i in range (4): corners.append(vd.stereo_triangulate_edges_centered(img_width, img_height, camFOV, T1, T2, corners_cam1[:,i], corners_cam2[:,i]))

            corners=np.squeeze(corners)

            A=corners[0:4,0:2]
            _,a=vd.order_rectangle_points(A)

            yaw =np.zeros(4)
            yaw[0]=(np.pi/4+a[2])
            yaw[1]=(-np.pi/4+a[3])
            yaw[2]= (np.pi*3/4+a[1])#kinematics.wrap_to_pi
            yaw[3]= (np.pi*5/4+a[0])
                        
            yaw_rect =np.mean(yaw)
            e_yaw=np.rad2deg(yaw_rect)-np.rad2deg(yaw_cube)
            e=np.transpose(point_3D)-env.pos_new_cube
    
            yaw_cube.append(yaw_rect)
            error_yaw.append(e_yaw)
            cal_3D.append(point_3D)
            act.append(env.pos_new_cube)
            error.append(e)
            step1+=1
            
            #if step1 == 1 or step1 == 5: #False:#
                #ctrl3.visualize_perception(img,edges,f_edges_p1,XY_new1,XY_p1,upper_edges1,polar1,polar_smt1,polar_all1,points1,hull1,best_name1)
                #ctrl3.visualize_perception(img2,edges2,f_edges_p2,XY_new2,XY_p2,upper_edges2,polar2,polar_smt2,polar_all2,points2,hull2,best_name2)
        if step1 == 5:
            break
        t += dt
        #print(t)
        t1+=dt
        step+=1
        
    #print(y)
    cal_3D=np.squeeze(cal_3D)
    len=np.shape(cal_3D)[0]
    n=np.arange(len)
    act=np.squeeze(act)
    act[:,2]=0.105*np.ones(np.shape(cal_3D)[0])
    P_Predict=np.zeros(3)
    P_Predict[0]=np.mean(cal_3D[:,0])
    P_Predict[1]=np.mean(cal_3D[-1,1])
    P_Predict[2]=np.mean(cal_3D[:,2])
    print("error in position : ",act[-1,:]-P_Predict)
    
    error=np.squeeze(error)
    yaw_cube=np.array([np.squeeze(yaw_cube)])

    
    yaw_mean=np.mean(yaw_cube)
    yaw_cube_act = env.cube_yaw
    
    err_yaw=abs(np.rad2deg(kinematics.wrap_to_pi(yaw_mean-yaw_cube_act)))
    tol=1e-3
    if err_yaw/90>1 or np.isclose(err_yaw,90):
        err_yaw=abs(err_yaw-90)
        
    print("error in orientation : ",err_yaw)
    pose_error=np.hstack([act[-1,:]-P_Predict,err_yaw])
    data = {
    "img": img,
    "edges": edges,
    "f_edges_p": f_edges_p1,
    "XY_new": XY_new1,
    "XY_p": XY_p1,
    "u_e1": upper_edges1,
    "polar": polar1,
    "polar_smt": polar_smt1,
    "polar_all": polar_all1,
    "points": points1,
    "hull": hull1,
    "best_name": best_name1,
    "pose_error": pose_error   # np.array([ex, ey, ez, eyaw])
    }


    return P_Predict,act,cal_3D,yaw_mean+np.pi/2,t, data

def cube_data(env,kinematics,q0,Pr0,z_off,t):
    
    # ================= CUBE DATA =================

    cube_pos,_,_,cube_yaw,t,data = cube_vision_data(env,t)#env.pos_new_cube #cube_yaw = env.cube_pose["yaw"]
    T_cube=kinematics.T_from_XYZ_RPY(np.hstack([cube_pos,0,0,0]))
    ctrl3.draw_frame(T_cube)
    
    belt_yaw = env.belt_orientation[-1]

    x0, y0, _ = cube_pos
    xr, yr = 0.0, 0.0
    R = 0.8

    X = line_circle_intersection(x0, y0, belt_yaw, xr, yr, R)
    if X is None: print("❌ No intersection found")
    X1, X2 = X
    print("Entry point: ",X1)
    print(" Exit point: ",X2)
    ctrl3.draw_frame(kinematics.T_from_XYZ_RPY([*X1, 0.07, 0, 0, 0]))
    ctrl3.draw_frame(kinematics.T_from_XYZ_RPY([*X2, 0.07, 0, 0, 0]))

    # ================= CUBE VELOCITY =================

    Vc = env.cube_velocity

    # ================= FIND CATCH POSE =================

    dq_max = np.ones(6) * 3.33

    target = find_optimal_catch_point_jacobian(env, kinematics, X1, X2, cube_pos,cube_yaw, Pr0, Vc, q0, dq_max, z_off)

    if target is None:
        print("⚠️ No feasible catch point")
        return

    # ================= IK FOR CATCH =================

    q_catch,_,_,_,_ = kinematics.ik_one(target, q0)

    if q_catch is None:
        print("❌ IK failed at catch pose")
        return
    
    ctrl3.draw_frame(kinematics.fk(q_catch))

    # ================= CATCH TIME =================

    t_catch = np.linalg.norm(target[:2] - cube_pos[:2]) / np.linalg.norm(Vc)
    print(f"⏱ Catch time: {t_catch:.3f} s")

    return target, q_catch,cube_yaw, t_catch, Vc , X2, t, data

# ==========================================================================================

# ========================= LINE TRAJECTORY TASK SPACE ==================================

def lineTrajectory(wait,dt,t_max_p,poe_d, v_cube,z_off,poe,p_d_prev,bb):

    if wait and poe[2] > poe_d and bb :

        Ve=np.array([0,v_cube[1],-(z_off-poe_d)/t_max_p])
        #grip=False
        
    else:

        Ve=np.array([0,v_cube[1],0])
        #grip=True

    p_d  = p_d_prev + Ve * dt
    dp_d = Ve#v_cube
    ddp_d = np.zeros(3)

    return p_d, dp_d, ddp_d

# ===================== CUBIC TRAJECTORY IN JOINT SPACE ====================================

def cubic_joint_trajectory(env, kinematics, t, t_catch, q0, dq0, q_catch, dq_max):

    J_full = kinematics.jacobian(q_catch)      # (6,6)
    damping=1e-6

    H = J_full.T @ J_full + damping * np.eye(6)
    g = J_full.T @ np.hstack([env.cube_velocity,env.cube_angular_velocity])
    dq_catch = np.clip((np.linalg.inv(H)@g),-dq_max,dq_max)
    
    q0 = np.asarray(q0, dtype=float)
    qf = np.asarray(q_catch, dtype=float)
    dqf = np.asarray(dq_catch, dtype=float)

    T = t_catch
    
    d = q0
    c = dq0
    a = (2*(d - qf) + (c + dqf)*T) / (T**3)
    b = (3*(qf - q0) - (2*dq0 + dqf)*T) / (T**2)

    q   = a * t**3 + b * t**2 +c*t + d 
    dq=np.clip(3 * a * t**2 + 2 * b * t + c,-dq_max,dq_max)
    ddq = 6 * a * t + 2 * b

    return q, dq, ddq

def cubic_joint_trajectory1(env, kinematics, t, t_catch, q0, dq0, q_catch, dq_max):
    dq_max=3.333

    dq_catch = np.zeros(6)
    q0 = np.asarray(q0, dtype=float)
    qf = np.asarray(q_catch, dtype=float)
    dqf = np.asarray(dq_catch, dtype=float)

    T = t_catch
    
    d = q0
    c = dq0
    a = (2*(d - qf) + (c + dqf)*T) / (T**3)
    b = (3*(qf - q0) - (2*dq0 + dqf)*T) / (T**2)

    q   = a * t**3 + b * t**2 +c*t + d 
    dq=np.clip(3 * a * t**2 + 2 * b * t + c,-dq_max,dq_max)
    ddq = 6 * a * t + 2 * b

    return q, dq, ddq

# ====================== Resolved Motion Rate Control (RMRC)  =================================
#                          IMPLEMENTATION FOR DES VELOCITY

def CLIK(env,kinematics, q0, q_catch, Ve):   

    J_full = kinematics.jacobian(q0)      # (6,6)
    dq_max=3.333
    damping=1e-6

    H = J_full.T @ J_full + damping * np.eye(6)
    dxd=np.hstack([Ve,env.cube_angular_velocity])
    xd=kinematics.pRPY(kinematics.fk(q_catch))
    x=kinematics.pRPY(kinematics.fk(q0))

    g = J_full.T @ (dxd)#+0.5*Kd@((xd-x)/t_catch1))

    dq_catch = np.clip((np.linalg.inv(H)@g),-dq_max,dq_max)
    qf = np.asarray(q_catch, dtype=float)
    dqf = np.asarray(dq_catch, dtype=float)

    q   = qf
    dq  = dqf
    dq  =np.clip(dq,-3.333,3.333)
    ddq = np.zeros(6)

    return q, dq, ddq
