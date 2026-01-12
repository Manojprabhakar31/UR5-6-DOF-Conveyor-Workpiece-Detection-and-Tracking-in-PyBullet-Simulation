import numpy as np
np.set_printoptions(precision=6, suppress=True)

class UR5_KINEMATICS:

# ===================== HELPER FUNCTIONS =========================================================
    
    @staticmethod
    def wrap_to_pi(angle):
        return (angle + np.pi) % (2*np.pi) - np.pi
    

    def __init__(self):

        # ========================= UR5 DH PARAMETERS ============================

        self.a = np.array([0, 0.425, 0.39225, 0, 0, 0], dtype=float)
        self.d = np.array([0.0892, 0, 0, 0.110, 0.09475, -0.1215], dtype=float)
        self.alpha = np.array([-np.pi/2, 0, 0, -np.pi/2, np.pi/2, np.pi], dtype=float)

        # ======================== JOINT LIMITS (rad) =======================

        limit = np.deg2rad(np.array([180,180,180,180,180,180]))
        self.q_min = -np.deg2rad(np.array([180,180,180,180,180,180]))#limit #* np.ones(6)
        self.q_max =  np.deg2rad(np.array([180,0.1,180,180,180,180]))#limit #* np.ones(6)

        # ======================== SINGULARITY EPS ===========================

        self.EPS_SHOULDER = 1e-6
        self.EPS_ELBOW    = 1e-6
        self.EPS_WRIST    = 1e-6

    # ====================== UR5 DH DATA ====================================

    def DH_table(self):
            a = np.array([0, 0.425, 0.39225, 0, 0, 0], dtype=float)
            d = np.array([0.0892, 0, 0, 0.110, 0.09475, -0.1215], dtype=float)
            alpha = np.array(
                [-np.pi/2, 0, 0, -np.pi/2, np.pi/2, np.pi],
                dtype=float
            )
            return a, d, alpha
    
    # ==================== T FROM DH DETAILS ====================================

    def DH(self, alpha, a, d, theta):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ ct, -st*ca,  st*sa, a*ct],
            [ st,  ct*ca, -ct*sa, a*st],
            [  0,     sa,     ca,    d],
            [  0,      0,      0,    1]
        ])

    # ================== FORWARD KINEMATICS ===================================

    def fk(self, q):
        T = np.eye(4)
        for i in range(6):
            T = T @ self.DH(self.alpha[i], self.a[i], self.d[i], q[i])
        return T

    # ===================== CONFIG FROM T ==============================

    def pRPY(self, T):
        p = T[0:3, 3]
        R = T[0:3, 0:3]

        s = np.hypot(R[0,0], R[1,0])
        pitch = np.arctan2(-R[2,0], s)

        if s > 1e-8:
            yaw  = np.arctan2(R[1,0], R[0,0])
            roll = np.arctan2(R[2,1], R[2,2])
        else:
            roll  = 0.0
            yaw   = np.arctan2(-R[0,1], R[1,1])
            pitch = np.sign(-R[2,0]) * np.pi / 2

        return np.array([
            p[0], p[1], p[2],
            self.wrap_to_pi(roll),
            self.wrap_to_pi(pitch),
            self.wrap_to_pi(yaw)
        ])

    # ================= T from CONFIG ===============================

    def T_from_XYZ_RPY(self, c):
        px, py, pz = c[0:3]
        r, p, y = np.deg2rad(c[3:6])

        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)

        R = np.array([
            [ cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [ sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [   -sp,          cp*sr,           cp*cr]
        ])

        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = [px, py, pz]
        return T

    # ===================================================
    # ==================== IK SOLVER ====================
    # ===================================================

    def ik_all(self, T06):
        p = T06[0:3, 3]
        R = T06[0:3, 0:3]

        theta1 = np.zeros(2)
        theta5 = np.zeros((2,2))
        theta6 = np.zeros((2,2))

        # ---- θ1 ----
        z6 = R[:,2]
        p5 = p - abs(self.d[5]) * z6
        r = np.hypot(p5[0], p5[1])

        if r < 1e-5:
            # Wrist center on base Z-axis → θ1 undefined
            # Choose valid representative solutions
            theta1[:] = [0.0, np.pi]
        else:
            phi = np.arctan2(p5[1], p5[0])
            delta = np.arccos(np.clip(self.d[3] / r, -1.0, 1.0))

        theta1[:] = [-np.pi/2 + phi + delta,
                     -np.pi/2 + phi - delta]

        # ---- θ5, θ6 ----
        for i in range(2):
            T01 = self.DH(self.alpha[0], self.a[0], self.d[0], theta1[i])
            T16 = np.linalg.solve(T01, T06)

            c5 = np.clip(-T16[2,2], -1, 1)
            s5 = np.sqrt(1 - c5**2)

            for j in range(2):
                theta5[i,j] = ((-1)**j) * np.arctan2(s5, c5)
                s5j = np.sin(theta5[i,j])
                if abs(s5j) < 1e-5:
                    # Wrist singularity: axes 4 and 6 aligned
                    # θ6 is free → choose representative values
                    theta6[j] = 0.0 if j == 0 else np.pi
                else:
                    
                    theta6[i,j] = 0.0 if abs(s5j) < self.EPS_WRIST else \
                              np.arctan2(-T16[2,1]/s5j, -T16[2,0]/s5j)

        # ---- θ2, θ3, θ4 ----
        sols = []
        for i in range(2):
            T01 = self.DH(self.alpha[0], self.a[0], self.d[0], theta1[i])
            T16 = np.linalg.solve(T01, T06)

            for j in range(2):
                T56 = self.DH(self.alpha[4], self.a[4], self.d[4], theta5[i,j]) @ \
                      self.DH(self.alpha[5], self.a[5], self.d[5], theta6[i,j])

                T14 = T16 @ np.linalg.inv(T56)
                p14 = T14[0:2,3]
                r3 = np.linalg.norm(p14)
                if r3 < 1e-6:
                    continue
                c3 = np.clip((r3**2 - self.a[1]**2 - self.a[2]**2) /
                             (2*self.a[1]*self.a[2]), -1, 1)

                for t3 in [np.arccos(c3), -np.arccos(c3)]:
                    t2 = -(np.arctan2(-p14[1], p14[0]) +
                           np.arcsin(self.a[2]/r3 * np.sin(t3)))

                    T13 = self.DH(self.alpha[1], self.a[1], self.d[1], t2) @ \
                          self.DH(self.alpha[2], self.a[2], self.d[2], t3)

                    T34 = np.linalg.solve(T13, T14)
                    t4 = np.arctan2(T34[1,0], T34[0,0])

                    sols.append([theta1[i], t2, t3, t4, theta5[i,j], theta6[i,j]])

        return np.array(sols)

    # ===================================================
    # ============ FILTER + BEST SOLUTION ===============
    # ===================================================

    def filter_joint_limits(self, qs):
        valid = []
        for q in qs:
            if np.all(q >= self.q_min) and np.all(q <= self.q_max):
                valid.append(q)
        return np.array(valid)

    def best_solution_fk_priority(self, qs, T_target, q_current, w_pos=1.0, w_rot=0.3, pose_tol=1e-4, joint_tol=1e-4):

        """
        Select best IK solution using:
        1) FK position + orientation error
        2) Joint displacement from current
        3) Random tie-break

        Returns:
            q_best           : selected joint configuration
            ee_pos           : achieved end-effector position (x,y,z)
            ee_rpy           : achieved end-effector orientation (roll,pitch,yaw) [rad]
            pos_err_best     : position error norm
            rot_err_best     : orientation error norm
        """

        if len(qs) == 0: return None, None, None, None, None

        pose_errors = []
        pos_errors  = []
        rot_errors  = []
        fk_cache    = []

        # ---------- Stage 1: FK pose error ----------

        for q in qs:
            T_fk = self.fk(q)
            fk_cache.append(T_fk)

            # Position error
            pos_err = np.linalg.norm(T_fk[:3, 3] - T_target[:3, 3])

            # Orientation error (Frobenius norm)
            rot_err = np.linalg.norm(T_fk[:3, :3] - T_target[:3, :3], ord='fro')

            pos_errors.append(pos_err)
            rot_errors.append(rot_err)

            pose_errors.append(w_pos * pos_err + w_rot * rot_err)

        pose_errors = np.array(pose_errors)
        pos_errors  = np.array(pos_errors)
        rot_errors  = np.array(rot_errors)

        min_pose_err = np.min(pose_errors)

        pose_idx = np.where(np.abs(pose_errors - min_pose_err) < pose_tol)[0]

        pose_candidates = qs[pose_idx]

        # ---------- Stage 2: Joint displacement ----------
        joint_errors = np.linalg.norm(pose_candidates - q_current, axis=1)

        min_joint_err = np.min(joint_errors)

        joint_idx = np.where(np.abs(joint_errors - min_joint_err) < joint_tol)[0]

        final_idx = pose_idx[joint_idx]

        # ---------- Stage 3: Random tie-break ----------
        best_id = final_idx[np.random.randint(len(final_idx))]

        q_best = qs[best_id]

        # ---------- Achieved EE pose ----------
        T_best = fk_cache[best_id]
        ee_cfg = self.pRPY(T_best)

        ee_pos = ee_cfg[0:3]
        ee_rpy = ee_cfg[3:6]

        pos_err_best = pos_errors[best_id]
        rot_err_best = rot_errors[best_id]
        
        return q_best, ee_pos, ee_rpy, pos_err_best, rot_err_best

    # ==================== OPTIMAL Q FROM CONFIG ===================================== 

    def ik_one(self,P_R_catch_3d, q_seed):

        T_target = self.T_from_XYZ_RPY(P_R_catch_3d)
        qs_all = self.ik_all(T_target)
        qs_valid = self.filter_joint_limits(qs_all)
        if len(qs_valid) == 0: return None

        return self.best_solution_fk_priority(qs_valid, T_target, q_seed)
    
    #============================ OFFSET =========================================

    def ik_to_urdf(self,q_ik):

        UR5_URDF_OFFSETS = np.array([
        0.0,        # shoulder_pan
        0.0,        # shoulder_lift
        0.0,        # elbow
        np.pi/2,    # wrist_1
        0.0,        # wrist_2
        0.0         # wrist_3
        ])
        return q_ik + UR5_URDF_OFFSETS
    
    # ======================== GEOMETRIC JACOBIAN ====================================

    def jacobian(self,q):

        a, d, alpha = self.DH_table()
        q = np.asarray(q)

        T = np.eye(4)
        p = [T[:3, 3]]
        z = [T[:3, 2]]

        for i in range(6):
            T = T @ self.DH(alpha[i], a[i], d[i], q[i])
            p.append(T[:3, 3])
            z.append(T[:3, 2])

        pe = p[-1]
        J = np.zeros((6, 6))

        for i in range(6):
            J[:3, i] = np.cross(z[i], pe - p[i])
            J[3:, i] = z[i]

        return J



# =======================================================
# ======================= DEMO ==========================
# =======================================================
if __name__ == "__main__":
    ur5 = UR5_KINEMATICS()

    # --------------------------------------------------
    # 1) Generate random reachable target pose
    # --------------------------------------------------
    max_reach = ur5.a[1] + ur5.a[2] + ur5.d[3]
    min_reach = 0.15

    r = np.random.uniform(min_reach, max_reach)
    theta = np.random.uniform(0, 2*np.pi)
    phi   = np.random.uniform(0, np.pi/2)

    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)

    roll  = np.random.uniform(-180, 180)
    pitch = np.random.uniform(-90, 90)
    yaw   = np.random.uniform(-180, 180)

    target_xyz_rpy = np.array([x, y, z, roll, pitch, yaw])#np.array([-0.912, 0.0135, 0.0892, 0, -90, -90])#
    T_target = ur5.T_from_XYZ_RPY(target_xyz_rpy)

    print("\nTarget XYZ RPY:", target_xyz_rpy)
    print("\nTarget Transform:\n", T_target)

    # --------------------------------------------------
    # 2) Compute all IK solutions (up to 8)
    # --------------------------------------------------
    qs_all = ur5.ik_all(T_target)
    print("\nAll IK solutions (deg):\n", np.rad2deg(qs_all))

    # --------------------------------------------------
    # 3) Filter by joint limits
    # --------------------------------------------------
    qs_valid = ur5.filter_joint_limits(qs_all)
    print("\nValid IK solutions (deg):\n", np.rad2deg(qs_valid))

    if len(qs_valid) == 0:
        print("\n❌ No valid IK solution within joint limits")
        exit()

    # --------------------------------------------------
    # 4) Select best solution using FK error
    # --------------------------------------------------
    q_current = np.zeros(6)   # reference posture (home)

    q_best = ur5.best_solution_fk_priority(
        qs_valid,
        T_target,
        q_current,
        w_rot=0.3
    )

    # --------------------------------------------------
    # 5) Display selected solution
    # --------------------------------------------------
    print("\n✅ Best IK solution (deg):\n", np.rad2deg(q_best[0]))

    # --------------------------------------------------
    # 6) FK verification
    # --------------------------------------------------
    T_fk = ur5.fk(q_best[0])

    pos_err = np.linalg.norm(
        T_fk[0:3, 3] - T_target[0:3, 3]
    )

    rot_err = np.linalg.norm(
        T_fk[0:3, :3] - T_target[0:3, :3],
        ord='fro'
    )

    print("\nFK of best solution:\n", T_fk)
    print("Position error:", pos_err)
    print("Orientation error (Frobenius):", rot_err)


    # ----------------- FK verification -----------------
    if q_best is not None:
        T_check = ur5.fk(q_best[0])
        print("\nFK of best IK solution:\n", T_check)
        print("Position error norm:", np.linalg.norm(T_target[0:3,3] - T_check[0:3,3]))

