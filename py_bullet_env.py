import pybullet as p
import pybullet_data
import numpy as np

class UR5BulletEnv:

    def __init__(self, gui=True):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.plane = p.loadURDF("plane.urdf")

        self.robot_id = self._load_robot()
        self._setup_camera()
        self._spawn_conveyor()
        self._spawn_tray()
        self.spawn_moving_cube_on_belt()
        
        #self.spawn_random_cube_on_belt()

    # ---------------- ROBOT ----------------

    def _load_robot(self):
        urdf_path = "urdf2/ur5_creo.urdf"
        start_pos = [0, 0, 0.0]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        
        return p.loadURDF(urdf_path, start_pos, start_ori)

    # ----------------- CUBE --------------------

    def spawn_random_cube(self):

        cube_size = 0.1
        belt_z = 0.02

        belt_length = 1.5
        belt_width  = 0.4

        r=np.random.uniform(0.4,0.8)
        theta=np.random.uniform(np.deg2rad(10),np.deg2rad(60))
        x=r*np.cos(theta)
        y=r*np.sin(theta)
        z = belt_z + cube_size / 2

        # ---------------- random yaw --------------------
        yaw = np.random.uniform(-np.pi, np.pi)
        orientation = p.getQuaternionFromEuler([0, 0, yaw])

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[cube_size/2]*3
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[cube_size/2]*3,
            rgbaColor=[0.9, 0.1, 0.1, 1]
        )
        #pos=np.array([0.486021,0.286722,0.17,0,0,np.deg2rad(-10.966242)])
        cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, z], #pos[0:3],#
            baseOrientation=orientation #pos[3:6]
        )

        self.cube_id = cube_id
        self.cube_pose = {
            "position": np.array([x, y, z]),#pos[0:3],#
            "orientation": np.array(orientation),#pos[3:6],#
            "yaw": yaw#pos[5]#
        }

        return cube_id, self.cube_pose

    def spawn_random_cube_on_belt(self):

        # ---------------- Cube parameters ----------------
        self.cube_size_r = 0.065
        cube_size_r = self.cube_size_r
        

        # ---------------- Belt parameters ----------------
        belt_length = 1.5
        belt_width  = 0.4
        belt_z      = 0.02

        belt_center_x = 0.6
        belt_center_y = 0.0

        # ---------------- Robot reach limits (UR5-safe) ----------------
        r_min = 0.35   # avoid too close to base
        r_max = 0.85   # keep margin from max reach

        # ---------------- Sample until valid ----------------
        max_trials = 100
        for _ in range(max_trials):

            # Sample on conveyor
            x = belt_center_x + np.random.uniform(
                -belt_width/2 + cube_size_r/2,
                belt_width/2 - cube_size_r/2
            )

            y = belt_center_y + np.random.uniform(
                -belt_length/2 + cube_size_r/2,
                belt_length/2 - cube_size_r/2
            )

            # Check robot reach
            r = np.sqrt(x**2 + y**2)

            if r_min <= r <= r_max:
                break
        else:
            raise RuntimeError("Failed to sample reachable cube position")

        # Height on belt
        z = belt_z + cube_size_r / 2

        # ---------------- Random yaw ----------------
        yaw = np.random.uniform(-np.pi, np.pi)
        orientation = p.getQuaternionFromEuler([0, 0, yaw])

        # ---------------- Collision & visual ----------------
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[cube_size_r / 2] * 3
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[cube_size_r / 2] * 3,
            rgbaColor=[0.9, 0.1, 0.1, 1]
        )
        self.cube_mass_r=2.0
        # ---------------- Create cube ----------------
        cube_id = p.createMultiBody(
            baseMass=self.cube_mass_r,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, z],
            baseOrientation=orientation
        )

        # ---------------- Store pose ----------------
        self.cube_id = cube_id
        self.cube_pose_r = {
            "position": np.array([x, y, z]),
            "orientation": np.array(orientation),
            "yaw": yaw
        }

        return cube_id, self.cube_pose_r

    def update_cube_on_belt(self, dt):
        if self.cube_id is None:
            return

        pos, orn = p.getBasePositionAndOrientation(self.cube_id)
        pos = np.array(pos)
        pos_new = pos + self.cube_velocity * dt
        orn_v = np.array(orn[0:3])
        orn_new = orn_v + self.cube_angular_velocity * dt
        self.pos_new_cube=pos_new
        self.orn_new_cube=orn_new
        p.resetBasePositionAndOrientation(
            self.cube_id,
            pos_new.tolist(),
            orn
        )

    def spawn_moving_cube_on_belt(self):

        cube_size = 0.065
        self.cube_size=cube_size
        belt_length = self.belt_length
        belt_width  = self.belt_width
        belt_height = self.belt_height
        position    = self.belt_position
        belt_center_x, belt_center_y = position[0:2]
        
        # ---- Spawn at belt entry ----
        x = belt_center_x + np.random.uniform(
            -belt_width/2 + cube_size*2,
            belt_width/2 - cube_size*2
        )

        y = belt_center_y - belt_length/2 + cube_size/2
        z = self.belt_position[2] + cube_size / 2

        # ---- Belt direction (same as conveyor) ----
        belt_orientation=self.belt_orientation
        belt_yaw = belt_orientation[-1] # 90 deg → +Y direction
        self.cube_yaw = np.random.uniform(-np.pi, np.pi)
        orientation = p.getQuaternionFromEuler([0, 0, self.cube_yaw])
        #print("self yaw of cube: ", self.cube_yaw)

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[cube_size/2]*3
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[cube_size/2]*3,
            rgbaColor=[0.9, 0.1, 0.1, 1]
        )
        self.cube_mass=2.0
        # ---- KINEMATIC cube ----
        cube_id = p.createMultiBody(
            baseMass=self.cube_mass,  # IMPORTANT: kinematic
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, z],
            baseOrientation=orientation
        )

        

        # ---- Conveyor velocity ----
        vc_c = 0.5  # m/s
        Vc = np.array([
            vc_c * np.cos(belt_yaw),
            vc_c * np.sin(belt_yaw),
            0.0
        ])
        self.cube_angular_velocity=np.array([0,0,0])
        # ---- Store for motion update ----
        self.cube_id = cube_id
        self.cube_velocity = Vc
        self.cube_pose = {
            "position": np.array([x, y, z]),
            "yaw": self.cube_yaw,
            "Vc": Vc
        }

        return cube_id, self.cube_pose


    # ---------------- CAMERA ----------------

    def _setup_camera(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=2,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0]
        )

    # ---------------- CONVEYOR ----------------

    def _spawn_conveyor(self):
        self.belt_length = 3.5
        self.belt_width  = 0.4
        self.belt_height = 0.01

        belt_length = self.belt_length
        belt_width  = self.belt_width
        belt_height = self.belt_height

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[belt_length/2, belt_width/2, belt_height/2]
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[belt_length/2, belt_width/2, belt_height/2],
            rgbaColor=[0.05, 0.05, 0.05, 1]
        )

        self.belt_position = [0.5, 0.0, belt_height/2]
        self.belt_orientation = [0, 0, 1.5707]
        position = self.belt_position
        orientation = p.getQuaternionFromEuler([0, 0, 1.5707])
        self.belt_id = p.createMultiBody(
            baseMass=2,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            baseOrientation=orientation
        )

    # ---------------- TRAY ----------------

    def _spawn_tray(self):
        tray_size = [0.5, 0.5, 0.001]

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[s/2 for s in tray_size]
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[s/2 for s in tray_size],
            rgbaColor=[0.3, 0.3, 0.3, 1]
        )

        position = [-0.6, -0.6, tray_size[2]/2]
        self.tray_position=position
        orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.tray_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            baseOrientation=orientation
        )
        
    # ---------------- STEP ----------------

    def step(self):
        
        dt = 1.0 / 420.0
        self.update_cube_on_belt(dt)
        p.stepSimulation()
    

    def step1(self):

        p.stepSimulation()

    def disconnect(self):
        p.disconnect()




