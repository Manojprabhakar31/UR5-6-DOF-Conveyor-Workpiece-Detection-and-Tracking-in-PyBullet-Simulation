import cv2
import pybullet as p
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt

def calculate_shape_properties(shape_type, points):
    """
    Returns aspect ratio, size, orientation, and center
    for a fitted shape.
    Orientation is in degrees (OpenCV convention).
    """

    result = {
        "aspect_ratio": 0.0,
        "size": {},
        "orientation_deg": None,
        "center": None
    }

    # Ensure correct shape and dtype
    points = np.asarray(points)

    if points.ndim == 3: points = points.squeeze()

    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3: return result

    points_cv = points.astype(np.float32)

    # =====================================================
    # RECTANGLE
    # =====================================================
    if shape_type == "Rectangle":
        rect = cv2.minAreaRect(points_cv)
        (cx, cy), (w, h), angle = rect

        if w == 0 or h == 0:
            return result

        aspect_ratio = max(w, h) / min(w, h)

        result["aspect_ratio"] = aspect_ratio
        result["size"] = {
            "width": w,
            "height": h,
            "area": w * h
        }
        result["orientation_deg"] = angle
        result["center"] = (cx, cy)

    # =====================================================
    # TRIANGLE
    # =====================================================
    elif shape_type == "Triangle":
        x, y, w, h = cv2.boundingRect(points_cv)

        if w == 0 or h == 0:
            return result

        aspect_ratio = h / w

        # Center from bounding rectangle
        cx = x + w / 2.0
        cy = y + h / 2.0

        # Orientation via PCA
        mean = np.mean(points_cv, axis=0)
        cov = np.cov((points_cv - mean).T)
        eigvals, eigvecs = np.linalg.eig(cov)
        main_axis = eigvecs[:, np.argmax(eigvals)]
        angle = np.degrees(np.arctan2(main_axis[1], main_axis[0]))

        result["aspect_ratio"] = aspect_ratio
        result["size"] = {
            "width": w,
            "height": h,
            "area": 0.5 * w * h   # approx
        }
        result["orientation_deg"] = angle
        result["center"] = (cx, cy)

    # =====================================================
    # CIRCLE
    # =====================================================
    elif shape_type == "Circle":
        (cx, cy), r = cv2.minEnclosingCircle(points_cv)

        result["aspect_ratio"] = 1.0
        result["size"] = {
            "radius": r,
            "diameter": 2 * r,
            "area": np.pi * r * r
        }
        result["orientation_deg"] = None  # undefined
        result["center"] = (cx, cy)
        
    return result

def order_rectangle_points(pts):
        pts = np.array(pts).reshape(4, 2)
        cx, cy = np.mean(pts, axis=0)
        ang = np.arctan2(pts[:,1] - cy, pts[:,0] - cx)
        return pts[np.argsort(ang)],ang[np.argsort(ang)]

def fitting_shape(u_e):
    if u_e.size > 0:

        hull = cv2.convexHull(u_e)
        hull_area = cv2.contourArea(hull)

        # Rectangle (RED)
        sq_data=cv2.boxPoints(cv2.minAreaRect(u_e.astype(np.float32)))
        rect_area = cv2.contourArea(sq_data)
        rect_pts,_ = order_rectangle_points(sq_data)

        rect_pts1=np.vstack([rect_pts,rect_pts[0,:]])
        
        # Triangle (ORANGE)
        tri_area, tri_pts = cv2.minEnclosingTriangle(u_e)
        ret, tri_raw = cv2.minEnclosingTriangle(u_e.astype(np.float32))
        tri = tri_raw.squeeze().astype(int)

        # Circle (YELLOW)
        circle_area = np.pi * cv2.minEnclosingCircle(u_e)[1]**2
        (cx, cy), r = cv2.minEnclosingCircle(u_e.astype(np.float32))
        circ = plt.Circle((cx, cy), r, edgecolor='yellow', facecolor='none', linewidth=1.3)
        
        areas = np.array([rect_area, tri_area, circle_area])
        ratios = hull_area / areas
        best = np.argmin(np.abs(1-ratios))
        
        best_name = ["Rectangle", "Triangle", "Circle"][best]

        # Mapping: Rectangle=3.0, Triangle=2.0, Circle=1.0
        global_shape_num = float([3.0, 2.0, 1.0][best])
        points = rect_pts1,tri,circ,r,cx,cy
        results = calculate_shape_properties(best_name, u_e)
        
    return best_name, results,points,hull

def rdp_opencv(points):
    """
    OpenCV RDP (approxPolyDP)
    """

    epsilon=0.5
    closed=True

    pts=np.transpose(points.astype(np.float32))
    pts1=pts.reshape(-1, 1, 2)
    #print("shape of points: ",np.shape(pts1))
    approx = cv2.approxPolyDP(pts1, epsilon, closed)

    return np.transpose(approx)

def process_image(colorImg, imgWidth, n, epsilon=2, minAreaRatio=0.01, cannyLow=10, cannyHigh=100):

    # -------------------------------
    # 1. Edge detection
    # -------------------------------
    gray = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=0.5)
    edges = cv2.Canny(gray_blur, cannyLow, cannyHigh)

    ys, xs = np.where(edges)
    edge_p = np.vstack((xs, ys))

    # -------------------------------
    # 2. ROI filtering (VECTORISED)
    # -------------------------------
    mask = ((edge_p[0] > 160) & (edge_p[0] < 475) & (edge_p[1] < 472))
    f_edges_p = edge_p[:, mask]

    if f_edges_p.shape[1] < 10: return edges, None, None, None, None, None, None, None

    # -------------------------------
    # 3. Centering
    # -------------------------------
    XY_new = np.mean(f_edges_p, axis=1, keepdims=True)
    f_edges_p_new = XY_new - f_edges_p

    # -------------------------------
    # 4. Polar transform
    # -------------------------------
    r = np.linalg.norm(f_edges_p_new, axis=0)
    theta = np.arctan2(f_edges_p_new[1], f_edges_p_new[0])

    # -------------------------------
    # 5. Angular binning
    # -------------------------------
    bins = np.linspace(-np.pi, np.pi, 60)#np.arange(-np.pi, np.pi, 5*np.pi/180)
    bin_idx = np.digitize(theta, bins) - 1
    n_bins = len(bins) - 1

    r_min, theta_min = [], []

    for k in range(n_bins):
        mask = bin_idx == k
        if np.any(mask):
            idx_local = np.argmin(r[mask])
            r_min.append(r[mask][idx_local])
            theta_min.append(theta[mask][idx_local])

    r_min = np.array(r_min)
    theta_min = np.array(theta_min)

    # -------------------------------
    # 6. Sort by angle
    # -------------------------------
    order = np.argsort(theta_min)
    r_min = r_min[order]
    theta_min = theta_min[order]

    # -------------------------------
    # 7. Spike smoothing + corner detection
    # -------------------------------

    for i in range(1, len(r_min)-1):

        if r_min[i+1]<r_min[i]>r_min[i-1] or r_min[i+1]>r_min[i]<r_min[i-1]:

            if abs(r_min[i] - r_min[i+1]) > 0.4 and abs(r_min[i] - r_min[i-1]) > 0.4:
                r_min[i] = 0.5 * (r_min[i+1] + r_min[i-1])

    r_min_smooth = savgol_filter(r_min, window_length=5, polyorder=3)

    
     
    # -------------------------------
    # 8. Back to Cartesian
    # -------------------------------
    x_min = -r_min_smooth * np.cos(theta_min)
    y_min = -r_min_smooth * np.sin(theta_min)

    upper_edges = np.vstack((x_min, y_min)) + XY_new
    #print(np.shape(upper_edges))
    upper_edges1 = np.squeeze(rdp_opencv(upper_edges))
    #print(np.shape(upper_edges1))
    # -------------------------------
    # 9. Shape fitting
    # -------------------------------
    best_name, results = None, None
    if upper_edges.shape[1] > 3: best_name, results,points,hull = fitting_shape(upper_edges.T.astype(np.int32))
    if upper_edges1.shape[1] > 3: best_name, results1,points1,hull1 = fitting_shape(upper_edges1.T.astype(np.float32))
    polar, polar_smt, polar_all = np.vstack((r_min, theta_min)), np.vstack((r_min_smooth, theta_min)), np.vstack((r, theta))
    return (edges, f_edges_p, results["center"] if results else None,results1["center"] if results1 else None,
        upper_edges1,
        polar,
        polar_smt, polar_all,
        points,hull,points1,hull1,
        best_name
    )


def stereo_triangulate_edges_centered(imgWidth, imgHeight, camFOV,
                                      T1,T2, p1, p2,
                                      matchTolerance=5.0):
    
    # ================ CAMERA INTRINSIC PROPERTIES ====================

    fovRad = np.radians(camFOV)
    fy = imgHeight / (2.0 * np.tan(fovRad / 2.0))
    fx = fy
    cx = imgWidth / 2.0
    cy = imgHeight / 2.0

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)
    #print(K)
    # --- Define camera positions
    
    Z1 = T1[:3,3]
    Z2 = T2[:3,3]
    Q1 = T1[:3,:3]
    Q2 = T2[:3,:3]

    # --- Back-project to camera-space rays to metrics
    #print(np.shape(p1))
    px1 = np.vstack([p1[0],p1[1],[1]])
    px2 = np.vstack([p2[0],p2[1],[1]])

    y_p_cs1 = np.linalg.inv(K) @ px1
    #print(y_p_cs1)
    y_p_cs2 = np.linalg.inv(K) @ px2
    #print(y_p_cs2)

        # --- Convert to world-space directions
    d1 = Q1 @ y_p_cs1
    d1 = d1 / np.linalg.norm(d1)
    d2 = Q2 @ y_p_cs2
    d2 = d2 / np.linalg.norm(d2)
    #print(np.shape(d1))
    #print(np.shape(d2))

    C1 = np.transpose(np.array([Z1.copy()]))
    C2 = np.transpose(np.array([Z2.copy()]))

    # --- Triangulate using least-distance method
    r = C2 - C1
    #r=np.transpose(r)
    #print(np.shape(r))
    c = (d1.T @ d2).item()
    e = (d1.T @ r).item()
    f = (d2.T @ r).item()
    den = 1.0 - c**2
    #print(c," ",e," ",f," ",den)
    lam1 = (e - c*f) / den
    lam2 = (-f + c*e) / den

    P1 = C1 + lam1 * d1
    #print(C1)
    #print(lam1*d1)
    P2 = C2 + lam2 * d2
    #print("P1: ", P1)
    #print("P2: ",P2)
    P_est = 0.5 * (P1 + P2)

    return P_est

def get_camera_image(camPos, camTarget, camUp, imgWidth, imgHeight, camFOV):

        viewMat = p.computeViewMatrix(cameraEyePosition=camPos, cameraTargetPosition=camTarget, cameraUpVector=camUp)
        projMat = p.computeProjectionMatrixFOV(camFOV, imgWidth/imgHeight, 0.01, 5.0)
        img = p.getCameraImage(imgWidth, imgHeight, viewMat, projMat)
        colorImg = np.reshape(img[2], (imgHeight, imgWidth, 4))[:, :, :3]
        #depthImg = np.reshape(img[3], (imgHeight, imgWidth))

        return colorImg, viewMat, projMat

def image_feed(camPos, camTarget):
    imgWidth, imgHeight = 640, 480
    camFOV = 60
    camUp = [0, 1, 0]
    img,viewMat,_= get_camera_image(camPos, camTarget, camUp, imgWidth, imgHeight, camFOV)
    T = np.array(viewMat, dtype=np.float64).reshape(4, 4, order='F')

    p.stepSimulation()
    
    return img,imgWidth,imgHeight,camFOV,T