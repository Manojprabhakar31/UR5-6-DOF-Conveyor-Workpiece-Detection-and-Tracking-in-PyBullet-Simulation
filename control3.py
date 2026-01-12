import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import cv2
from kinematics import UR5_KINEMATICS
kinematics=UR5_KINEMATICS()

# =========================== PERCEPTION =========================================

def visualize_perception(
    img,
    edges,
    f_edges_p,
    XY_new,
    XY_p,
    u_e1,
    polar,
    polar_smt,
    polar_all,
    points,
    hull,
    best_name
):
    """
    Visualization of perception pipeline:
    - Edge extraction
    - Filtered edge points
    - Polar representation
    - Global shape fitting
    """

    rect_pts, tri, circ, r, cx, cy = points
    u_e = u_e1.T if u_e1.size > 0 else None

    fig, axs = plt.subplots(2, 3, figsize=(16, 10))

    # ==========================================================
    # 1. Raw Edge Map (Pixel Space)
    # ==========================================================
    ys, xs = np.where(edges)
    ax = axs[0, 0]
    ax.scatter(xs, ys, s=1)
    ax.set_title("1. Raw Edge Pixels")
    ax.set_aspect('equal')
    ax.grid(True)

    # ==========================================================
    # 2. Filtered & Processed Edge Points (XY Space)
    # ==========================================================
    ax = axs[0, 1]
    ax.scatter(f_edges_p[0], f_edges_p[1], s=2, color='red', label='Filtered Edges')
    ax.scatter(XY_new[0], XY_new[1], s=2, color='blue', label='Resampled Points')

    if u_e is not None:
        ax.scatter(u_e[:, 0], u_e[:, 1], s=5, color='black', label='Extreme Points')

    # Draw best-fit shape
    if best_name == "Rectangle":
        ax.plot(rect_pts[:, 0], rect_pts[:, 1], 'g', lw=1.0, label='Rectangle Fit')

    elif best_name == "Triangle":
        ax.plot(
            [*tri[:, 0], tri[0, 0]],
            [*tri[:, 1], tri[0, 1]],
            'r', lw=1.5, label='Triangle Fit'
        )

    elif best_name == "Circle":
        ax.add_patch(plt.Circle((cx, cy), r, fill=False,
                                 edgecolor='yellow', lw=1.5, label='Circle Fit'))

    ax.set_title("2. Filtered Edge Points & Best Fit")
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend(fontsize=8)

    # ==========================================================
    # 3. Polar Representation
    # ==========================================================
    ax = axs[0, 2]
    ax.scatter(polar_all[1], polar_all[0], s=1, color='red', label='Raw Polar')
    ax.plot(polar[1], polar[0], lw=1.5, label='Polar')
    ax.plot(polar_smt[1], polar_smt[0], lw=1.5, color='green', linestyle='--', label='Smoothed Polar')

    ax.set_xlim(-3.16, 3.16)
    ax.set_title("3. Polar Space Representation")
    ax.grid(True)
    ax.legend(fontsize=8)

    # ==========================================================
    # 4. Global Fits on Image
    # ==========================================================
    ax = axs[1, 0]
    ax.imshow(img)
    ax.set_title("4. All Global Shape Fits")

    # Rectangle
    ax.plot(
        [*rect_pts[:, 0], rect_pts[0, 0]],
        [*rect_pts[:, 1], rect_pts[0, 1]],
        'g', lw=1.3
    )

    # Triangle
    ax.plot(
        [*tri[:, 0], tri[0, 0]],
        [*tri[:, 1], tri[0, 1]],
        'r', lw=1.3
    )

    # Circle
    ax.add_patch(circ)

    ax.axis('off')

    # ==========================================================
    # 5. Best Global Fit Only
    # ==========================================================
    ax = axs[1, 1]
    ax.imshow(img)
    ax.set_title(f"5. Best Fit: {best_name}")

    if best_name == "Rectangle":
        ax.plot(
            [*rect_pts[:, 0], rect_pts[0, 0]],
            [*rect_pts[:, 1], rect_pts[0, 1]],
            'g', lw=1.5
        )

    elif best_name == "Triangle":
        ax.plot(
            [*tri[:, 0], tri[0, 0]],
            [*tri[:, 1], tri[0, 1]],
            'r', lw=1.5
        )

    elif best_name == "Circle":
        ax.add_patch(plt.Circle((cx, cy), r, fill=False,
                                 edgecolor='yellow', lw=1.5))

    ax.axis('off')

    # ==========================================================
    # 6. Empty / Reserved
    # ==========================================================
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()



# ============================ CONTROL ================================================

# ============================= DRAW FRAME FROM T ====================================

def draw_frame(T, axis_len=0.15, line_width=3, life_time=0):

    p0 = T[:3, 3]
    x_axis = p0 + axis_len * T[:3, 0]
    y_axis = p0 + axis_len * T[:3, 1]
    z_axis = p0 + axis_len * T[:3, 2]

    p.addUserDebugLine(p0, x_axis, [1, 0, 0], line_width, life_time)
    p.addUserDebugLine(p0, y_axis, [0, 1, 0], line_width, life_time)
    p.addUserDebugLine(p0, z_axis, [0, 0, 1], line_width, life_time)

def wrap_to_pi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

#===============================================================================

# ============================== PLOTS ======================================================

def plot_all(env, p_des_log, p_act_log, q_log2, tau_log2,
             t_log1,t_log2, cube_log, q_catch, e_yaw,res1, pc):
    t_log  =np.hstack([t_log1,t_log2])
    labels = ['X', 'Y', 'Z', 'Yaw']

    # ==========================================================
    # 1. JOINT-SPACE ANALYSIS
    # ==========================================================
    tau_log1,q_log1,q_err1=np.array([res1["torque"]]), np.array([res1["joints"]]), np.array([res1["error"]])
    
    tau_log1=np.squeeze(tau_log1).T
    q_log1=np.squeeze(q_log1).T
    q_err1=np.squeeze(q_err1).T
    print(np.shape(tau_log1)," ",np.shape(tau_log2))
    q_catch_off = np.squeeze([kinematics.ik_to_urdf(q) for q in q_catch])
    q_err2 = q_log2 - q_catch_off
    q_log=np.vstack([q_log1,q_log2])
    tau_log=np.vstack([tau_log1,tau_log2])
    q_err = np.vstack([q_err1,q_err2])
    fig, axs = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    for i in range(6):
        axs[0].plot(t_log, tau_log[:, i], label=f'J{i+1}')
        axs[1].plot(t_log, q_log[:, i], label=f'J{i+1}')
        axs[2].plot(t_log, np.rad2deg(q_err[:, i]), label=f'J{i+1}')
    #axs[0].axvline(t_log1[-1],"k:")
    axs[0].set_ylabel('Torque (N·m)')
    axs[0].set_title('Joint Torques')
    axs[0].axvline(t_log1[-1],color='k', linestyle='--',linewidth=1)
    axs[1].set_ylabel('Joint Angle (rad)')
    axs[1].set_title('Joint Positions')
    axs[1].axvline(t_log1[-1],color='k', linestyle='--',linewidth=1)
    axs[2].axhline(0, color='k', ls='--', alpha=0.5)
    axs[2].set_ylabel('Joint Error (deg)')
    axs[2].set_title('Joint Error w.r.t Catch Configuration')
    axs[2].set_xlabel('Time (s)')
    axs[2].axvline(t_log1[-1],color='k', linestyle='--',linewidth=1)

    for ax in axs:
        ax.grid(True)
        ax.legend(ncol=3, fontsize=9)

    fig.suptitle('Joint-Space Behavior', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


    
    # ==========================================================
    # 2. TASK-SPACE POSITION & YAW
    # ==========================================================
    fig, axs = plt.subplots(4, 1, figsize=(13, 9), sharex=True)

    # --- Position (X, Y, Z) ---
    for i, ax in enumerate(axs[:3]):
        ax.plot(t_log2, p_des_log[:, i], 'b--', lw=2, label='EE Desired')
        ax.plot(t_log2, p_act_log[:, i], 'r',  lw=2, label='EE Actual')
        ax.plot(t_log2, cube_log[:, i], 'g--', lw=2, label='Cube')
        ax.set_ylabel(f'{labels[i]} (m)')
        ax.set_title(f'End-Effector {labels[i]} Position')
        ax.grid(True)
        ax.legend(loc='best')

    # --- Yaw ---
    cube_yaw = env.cube_yaw
    yaw_err = e_yaw - cube_yaw

    axs[3].plot(t_log2, np.rad2deg(e_yaw), 'r', lw=2, label='EE Yaw')
    axs[3].plot(t_log2, np.rad2deg(cube_yaw) * np.ones_like(t_log2),
                'g--', lw=2, label='Cube Yaw')
    axs[3].plot(t_log2, np.rad2deg(yaw_err), 'k--', lw=2, label='Yaw Error')

    axs[3].set_ylabel('Yaw (rad)')
    axs[3].set_title('End-Effector Yaw Tracking')
    axs[3].grid(True)
    axs[3].legend(loc='best')

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle('Task-Space Tracking: Position and Orientation', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # ==========================================================
    # 3. 3D END-EFFECTOR TRAJECTORY
    # ==========================================================
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(p_des_log[:,0],p_des_log[:,1],p_des_log[:,2], 'r', lw=2, label='Desired EE Trajectory')

    key_points = {
        'Start': (p_des_log[0], 'blue'),
        'End':   (p_des_log[-1], 'green'),
        'Catch': (pc, 'black')
    }

    for label, (p, color) in key_points.items():
        ax.scatter(*p, color=color, s=80, label=label)

    ax.plot(*np.vstack((p_des_log[0], p_des_log[-1])).T,
            'k--', alpha=0.4, label='Straight-Line Reference')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Desired End-Effector Trajectory in Task Space')

    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
