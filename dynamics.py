import numpy as np
import math

# ======================================================
# MASS MATRICES
# ======================================================

def mass_matrix_th1_th4(
    th2, th3, th4,
    m1, m2, m3, m4,
    l2, l3, l4,
    r1, r2, r3, r4
):
    M = np.zeros((4, 4))

    # ---------------- M[0,0] ----------------
    M[0, 0] = (
        (l2**2*m2)/8 + (l2**2*m3)/2 + (l2**2*m4)/2
        + (l3**2*m3)/8 + (l3**2*m4)/2
        + (l4**2*m4)/8
        + (m1*r1**2)/2

        + (l2**2*m2*math.cos(2*th2))/8
        + (l2**2*m3*math.cos(2*th2))/2
        + (l2**2*m4*math.cos(2*th2))/2

        + (l3**2*m3*math.cos(2*(th2+th3)))/8
        + (l3**2*m4*math.cos(2*(th2+th3)))/2

        + (l4**2*m4*math.cos(2*(th2+th3+th4)))/8

        + (l2*l3*m3*math.cos(th3))/2
        + l2*l3*m4*math.cos(th3)

        + (l2*l3*m3*math.cos(2*th2+th3))/2
        + l2*l3*m4*math.cos(2*th2+th3)

        + (l2*l4*m4*math.cos(th3+th4))/2
        + (l2*l4*m4*math.cos(2*th2+th3+th4))/2

        + (l3*l4*m4*math.cos(th4))/2
        + (l3*l4*m4*math.cos(2*th2+2*th3+th4))/2
    )

    # ---------------- M[1,1] ----------------
    M[1, 1] = (
        (l2**2*m2)/3 + l2**2*m3 + l2**2*m4
        + (l3**2*m3)/3 + l3**2*m4
        + (l4**2*m4)/3
        + (m2*r2**2)/4 + (m3*r3**2)/4 + (m4*r4**2)/4
        + l2*l3*m3*math.cos(th3)
        + 2*l2*l3*m4*math.cos(th3)
        + l2*l4*m4*math.cos(th3+th4)
        + l3*l4*m4*math.cos(th4)
    )

    # ---------------- M[1,2] ----------------
    M[1, 2] = (
        (l3**2*m3)/3 + l3**2*m4
        + (l4**2*m4)/3
        + (m3*r3**2)/4 + (m4*r4**2)/4
        + (l2*l3*m3*math.cos(th3))/2
        + l2*l3*m4*math.cos(th3)
        + (l2*l4*m4*math.cos(th3+th4))/2
        + l3*l4*m4*math.cos(th4)
    )

    # ---------------- M[1,3] ----------------
    M[1, 3] = m4 * (
        4*l4**2 + 3*r4**2
        + 6*l3*l4*math.cos(th4)
        + 6*l2*l4*math.cos(th3+th4)
    ) / 12

    # ---------------- M[2,2] ----------------
    M[2, 2] = (
        (l3**2*m3)/3 + l3**2*m4
        + (l4**2*m4)/3
        + (m3*r3**2)/4 + (m4*r4**2)/4
        + l3*l4*m4*math.cos(th4)
    )

    # ---------------- M[2,3] ----------------
    M[2, 3] = m4 * (
        4*l4**2 + 6*l3*l4*math.cos(th4) + 3*r4**2
    ) / 12

    # ---------------- M[3,3] ----------------
    M[3, 3] = m4 * (4*l4**2 + 3*r4**2) / 12

    # ----------- symmetry -----------
    M[2, 1] = M[1, 2]
    M[3, 1] = M[1, 3]
    M[3, 2] = M[2, 3]

    return M


def mass_matrix_th5_th6(m4, m5, m6, l5, l6, r4, r5, r6):
    M11 = (
        (l5**2*m5)/3 + l5**2*m6 + (l6**2*m6)/3
        + (m4*r4**2)/2
        + (m5*r5**2)/4
        + (3*m6*r6**2)/4
        + l5*l6*m6
    )
    M12 = (m6*r6**2)/2
    M22 = (m6*r6**2)/2

    return np.array([[M11, M12],
                     [M12, M22]])


# ======================================================
# CORIOLIS MATRICES
# ======================================================

def coriolis_matrix_th1_th4(
    th2, th3, th4,
    dth1, dth2, dth3, dth4,
    m2, m3, m4,
    l2, l3, l4
):
    C = np.zeros((4, 4))

    # ================= C[0,*] =================
    C[0, 0] = (
        -dth2 * (
            (l3**2*m3*math.sin(2*th2 + 2*th3))/8
            + (l3**2*m4*math.sin(2*th2 + 2*th3))/2
            + (l2**2*m2*math.sin(2*th2))/8
            + (l2**2*m3*math.sin(2*th2))/2
            + (l2**2*m4*math.sin(2*th2))/2
            + (l4**2*m4*math.sin(2*th2 + 2*th3 + 2*th4))/8
            + (l2*l4*m4*math.sin(2*th2 + th3 + th4))/2
            + (l3*l4*m4*math.sin(2*th2 + 2*th3 + th4))/2
            + (l2*l3*m3*math.sin(2*th2 + th3))/2
            + l2*l3*m4*math.sin(2*th2 + th3)
        )
        -dth3 * (
            (l3**2*m3*math.sin(2*th2 + 2*th3))/8
            + (l3**2*m4*math.sin(2*th2 + 2*th3))/2
            + (l4**2*m4*math.sin(2*th2 + 2*th3 + 2*th4))/8
            + (l2*l4*m4*math.sin(2*th2 + th3 + th4))/4
            + (l2*l4*m4*math.sin(th3 + th4))/4
            + (l2*l3*m3*math.sin(th3))/4
            + (l2*l3*m4*math.sin(th3))/2
            + (l3*l4*m4*math.sin(2*th2 + 2*th3 + th4))/2
            + (l2*l3*m3*math.sin(2*th2 + th3))/4
            + (l2*l3*m4*math.sin(2*th2 + th3))/2
        )
        - (dth4*l4*m4*(
            l4*math.sin(2*th2 + 2*th3 + 2*th4)
            + 2*l2*math.sin(2*th2 + th3 + th4)
            + 2*l2*math.sin(th3 + th4)
            + 2*l3*math.sin(th4)
            + 2*l3*math.sin(2*th2 + 2*th3 + th4)
        ))/8
    )

    C[0, 1] = -dth1 * (
        (l3**2*m3*math.sin(2*th2 + 2*th3))/8
        + (l3**2*m4*math.sin(2*th2 + 2*th3))/2
        + (l2**2*m2*math.sin(2*th2))/8
        + (l2**2*m3*math.sin(2*th2))/2
        + (l2**2*m4*math.sin(2*th2))/2
        + (l4**2*m4*math.sin(2*th2 + 2*th3 + 2*th4))/8
        + (l2*l4*m4*math.sin(2*th2 + th3 + th4))/2
        + (l3*l4*m4*math.sin(2*th2 + 2*th3 + th4))/2
        + (l2*l3*m3*math.sin(2*th2 + th3))/2
        + l2*l3*m4*math.sin(2*th2 + th3)
    )

    C[0, 2] = -dth1 * (
        (l3**2*m3*math.sin(2*th2 + 2*th3))/8
        + (l3**2*m4*math.sin(2*th2 + 2*th3))/2
        + (l4**2*m4*math.sin(2*th2 + 2*th3 + 2*th4))/8
        + (l2*l4*m4*math.sin(2*th2 + th3 + th4))/4
        + (l2*l4*m4*math.sin(th3 + th4))/4
        + (l2*l3*m3*math.sin(th3))/4
        + (l2*l3*m4*math.sin(th3))/2
        + (l3*l4*m4*math.sin(2*th2 + 2*th3 + th4))/2
        + (l2*l3*m3*math.sin(2*th2 + th3))/4
        + (l2*l3*m4*math.sin(2*th2 + th3))/2
    )

    C[0, 3] = -(dth1*l4*m4*(
        l4*math.sin(2*th2 + 2*th3 + 2*th4)
        + 2*l2*math.sin(2*th2 + th3 + th4)
        + 2*l2*math.sin(th3 + th4)
        + 2*l3*math.sin(th4)
        + 2*l3*math.sin(2*th2 + 2*th3 + th4)
    ))/8

    # ================= C[1,*] =================
    C[1, 0] = -C[0, 1]

    C[1, 1] = (
        - dth4*((l2*l4*m4*math.sin(th3 + th4))/2 + (l3*l4*m4*math.sin(th4))/2)
        - dth3*((l2*l4*m4*math.sin(th3 + th4))/2
                + (l2*l3*m3*math.sin(th3))/2
                + l2*l3*m4*math.sin(th3))
    )

    C[1, 2] = (
        -(dth2*l2*(l3*m3*math.sin(th3)
                   + 2*l3*m4*math.sin(th3)
                   + l4*m4*math.sin(th3 + th4)))/2
        -(dth3*l2*(l3*m3*math.sin(th3)
                   + 2*l3*m4*math.sin(th3)
                   + l4*m4*math.sin(th3 + th4)))/2
        -(dth4*l4*m4*(l2*math.sin(th3 + th4)
                       + l3*math.sin(th4)))/2
    )

    C[1, 3] = -(l4*m4*(l2*math.sin(th3 + th4)
                       + l3*math.sin(th4))*(dth2 + dth3 + dth4))/2

    # ================= C[2,*] =================
    C[2, 0] = -C[0, 2]

    C[2, 1] = (
        dth2*((l2*l4*m4*math.sin(th3 + th4))/2
              + (l2*l3*m3*math.sin(th3))/2
              + l2*l3*m4*math.sin(th3))
        - (dth4*l3*l4*m4*math.sin(th4))/2
    )

    C[2, 2] = -(dth4*l3*l4*m4*math.sin(th4))/2

    C[2, 3] = -(l3*l4*m4*math.sin(th4)*(dth2 + dth3 + dth4))/2

    # ================= C[3,*] =================
    C[3, 0] = -C[0, 3]

    C[3, 1] = (
        dth2*((l2*l4*m4*math.sin(th3 + th4))/2
              + (l3*l4*m4*math.sin(th4))/2)
        + (dth3*l3*l4*m4*math.sin(th4))/2
    )

    C[3, 2] = (l3*l4*m4*math.sin(th4)*(dth2 + dth3))/2
    C[3, 3] = 0.0

    return C

def coriolis_matrix_th5_th6():
    return np.zeros((2, 2))


# ======================================================
# GRAVITY VECTORS
# ======================================================

def gravity_vector_th1_th4(
    th2, th3, th4,
    l2, l3, l4,
    m2, m3, m4,
    g
):
    """
    Gravity vector G(q) for yaw + 3-link pendulum
    Exact match to symbolic formulation
    """

    G = np.zeros((4,1))

    # G1 = 0 (yaw joint)
    G[0] = 0.0

    # G2
    G[1] = -g * (
        m4 * (
            l3 * np.cos(th2 + th3)
            + l2 * np.cos(th2)
            + (l4 * np.cos(th2 + th3 + th4)) / 2
        )
        + m3 * (
            (l3 * np.cos(th2 + th3)) / 2
            + l2 * np.cos(th2)
        )
        + (l2 * m2 * np.cos(th2)) / 2
    )

    # G3
    G[2] = -g * (
        m4 * (
            l3 * np.cos(th2 + th3)
            + (l4 * np.cos(th2 + th3 + th4)) / 2
        )
        + (l3 * m3 * np.cos(th2 + th3)) / 2
    )

    # G4
    G[3] = -(g * l4 * m4 * np.cos(th2 + th3 + th4)) / 2

    return G


def gravity_vector_th5_th6():
    import numpy as np

def gravity_term_th5(
    th2, th3, th4, th5,
    l4, l5, l6,
    m5, m6,
    g=9.81
):
    th234 = th2 + th3 + th4

    # ----- Denominators -----
    den_5 = np.sqrt((l5**2 * np.sin(th5)**2) / 4 + l4**2)
    den_6 = np.sqrt((np.sin(th5)**2) * (l5 + l6/2)**2 + l4**2)

    # ----- Term from link 5 -----
    term_5 = (
        l5**2 * m5
        * np.sin(th234 + np.arctan2(l5/2, l4))
        * np.cos(th5) * np.sin(th5)
    ) / (4 * den_5)

    # ----- Term from link 6 -----
    term_6 = (
        m6 * (l5 + l6/2)**2
        * np.cos(th5) * np.sin(th5)
        * np.sin(th234 + np.arctan2(l5 + l6/2, l4))
    ) / den_6

    # ----- Final gravity term -----
    G5 = -g * (term_5 + term_6)

    #return G5
    G = np.vstack([G5,0])
    return G #np.zeros((2,1))


# ======================================================
# FINAL 6-DOF ASSEMBLY
# ======================================================

def compute_MCG_6dof(q, dq, p):
    q  = np.asarray(q)
    dq = np.asarray(dq)

    th1, th2, th3, th4, th5, th6 = q
    dth1, dth2, dth3, dth4, dth5, dth6 = dq

    M14 = mass_matrix_th1_th4(
        th2, th3, th4,
        p["m1"], p["m2"], p["m3"], p["m4_1"],
        p["l1"], p["l2"], p["l3"],
        p["r1"], p["r2"], p["r3"], p["r4"]
    )

    M56 = mass_matrix_th5_th6(
        p["m4_2"], p["m5"], p["m6"],
        p["l5"], p["l6"],
        p["r4"], p["r5"], p["r6"]
    )

    C14 = coriolis_matrix_th1_th4(
        th2, th3, th4,
        dth1, dth2, dth3, dth4,
        p["m2"], p["m3"], p["m4_1"],
        p["l2"], p["l3"], p["l4"]
    )

    C56 = coriolis_matrix_th5_th6()

    G14 = gravity_vector_th1_th4(
        th2, th3, th4,
        p["l2"], p["l3"], p["l4"],
        p["m2"], p["m3"], p["m4_1"],
        p["g"]
    )

    G56 = gravity_term_th5(
        th2, th3, th4, th5,
        p["l4"], p["l5"], p["l6"],
        p["m5"],p["m6"]
)
    #gravity_vector_th5_th6()

    M = np.block([[M14, np.zeros((4,2))],
                  [np.zeros((2,4)), M56]])

    C = np.block([[C14, np.zeros((4,2))],
                  [np.zeros((2,4)), C56]])

    G = np.vstack([G14, G56])

    return M, C, G


if __name__ == "__main__":
    import numpy as np

    # ----------------- Joint states -----------------
    q  = [0.2, 0.4, -0.3, 0.1, 0.1, 0.1]   # [th1, th2, th3, th4, th5, th6]
    dq = [0.1, 0.05, -0.02, 0.01, 0.01, 0.01] # [dth1, dth2, dth3, dth4, dth5, dth6]

    # ----------------- Physical parameters -----------------
    params = {
        "g": 9.81,
        "m1": 11.861916, "m2": 27.490578, "m3": 19.185826, "m4_1": 22.3843771,
        "m4_2": 3.1985511, "m5": 3.2763593, "m6": 1.84699638, "mp": 0.70443738,     #m6 1.1425590
        "l1": 0.425, "l2": 0.39225, "l3": 0.09475, "l4": 0.047375,
        "l5": 0.04625, "l6": 0.07725,
        "r0": 0.116, "r1": 0.090, "r2": 0.075, "r3": 0.075,
        "r4": 0.075, "r5": 0.075, "r6": 0.075
    }

    # ----------------- Compute full M, C, G -----------------
    M, C, G = compute_MCG_6dof(q, dq, params)
    
    # ----------------- Print results -----------------
    np.set_printoptions(precision=4, suppress=True)
    print("\n==== Mass Matrix M (6x6) ====")
    print(M)
    print(np.linalg.inv(M))
    print("\n==== Coriolis Matrix C (6x6) ====")
    print(C)

    print("\n==== Gravity Vector G (6x1) ====")
    print(G)

    # ----------------- Check symmetry -----------------
    sym_error = np.linalg.norm(M - M.T)
    print("\nSymmetry error ||M - M.T|| =", sym_error)

    # ----------------- Check positive definiteness -----------------
    eigenvalues = np.linalg.eigvals(M)
    print("Eigenvalues of M:", eigenvalues)
    if np.all(eigenvalues > 0):
        print("✅ Mass matrix is POSITIVE DEFINITE")
    else:
        print("❌ Mass matrix is NOT positive definite")
