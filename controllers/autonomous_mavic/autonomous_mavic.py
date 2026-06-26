from controller import Robot
import sys
import random
import optparse
try:
    import numpy as np
    from numpy import NaN, nan
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")
try:
    import cv2
except ImportError:
    sys.exit("Warning: 'cv2' module not found.")


# ---------------------------------------------------------------------------
# Terrain elevation grid — extracted from forest_firefighters.wbt.
# ElevationGrid: 20×20 points, spacing 1.5789 m.
# The Transform has a 120° rotation around axis (1,1,1) which cycles axes:
#   world X ← local Z  →  iz = world_x / spacing
#   world Y ← local X  →  ix = world_y / spacing
#   world Z ← local Y  →  the height values themselves
# So _TERRAIN_H[iz, ix] gives the terrain world-Z at that grid cell.
# ---------------------------------------------------------------------------
_TERRAIN_X_DIM = 20
_TERRAIN_Z_DIM = 20
_TERRAIN_SPACING = 1.5789473684210527

_TERRAIN_H = np.array([
    # iz= 0 (world X ≈  0.0 m)
    [15.000, 13.441, 12.864, 12.500, 12.688, 12.488, 12.010, 12.679, 14.012, 14.549, 13.621, 12.880, 12.825, 13.232, 14.732, 15.501, 16.449, 17.749, 17.174, 16.259],
    # iz= 1 (world X ≈  1.6 m)
    [17.125, 15.666, 15.417, 15.070, 15.014, 14.710, 14.188, 14.496, 15.481, 16.153, 15.492, 15.085, 15.396, 15.466, 16.290, 16.781, 17.397, 18.465, 18.260, 17.830],
    # iz= 2 (world X ≈  3.2 m)
    [18.176, 16.646, 16.589, 16.307, 16.479, 16.289, 16.069, 16.476, 16.767, 17.404, 17.104, 16.661, 16.531, 16.284, 17.138, 17.590, 17.690, 18.940, 19.779, 19.430],
    # iz= 3 (world X ≈  4.7 m)
    [18.879, 17.072, 16.718, 16.698, 17.722, 17.947, 17.838, 18.292, 18.408, 18.159, 17.479, 17.115, 17.176, 17.094, 18.001, 18.244, 17.950, 18.990, 20.185, 20.297],
    # iz= 4 (world X ≈  6.3 m)
    [19.535, 17.369, 16.279, 16.791, 17.851, 18.391, 18.901, 19.271, 19.341, 18.557, 18.152, 18.319, 18.790, 19.042, 19.091, 19.381, 19.344, 19.386, 20.211, 20.361],
    # iz= 5 (world X ≈  7.9 m)
    [19.197, 17.269, 16.596, 17.251, 17.347, 17.623, 18.510, 18.978, 19.430, 19.715, 20.069, 20.077, 19.890, 20.221, 20.203, 20.748, 21.031, 20.709, 20.962, 20.676],
    # iz= 6 (world X ≈  9.5 m)
    [18.914, 17.078, 16.913, 17.290, 16.525, 16.491, 17.602, 18.485, 19.411, 20.845, 21.743, 21.480, 20.807, 21.178, 21.170, 21.620, 21.940, 21.681, 21.497, 20.861],
    # iz= 7 (world X ≈ 11.1 m)
    [19.633, 17.262, 16.441, 16.254, 15.583, 15.878, 17.358, 18.593, 19.582, 20.825, 21.798, 21.970, 22.024, 22.576, 22.251, 22.383, 22.309, 21.431, 20.859, 19.882],
    # iz= 8 (world X ≈ 12.6 m)
    [19.606, 17.208, 15.726, 15.065, 14.935, 15.871, 17.558, 18.632, 19.281, 19.660, 20.475, 21.073, 21.502, 22.110, 22.321, 22.799, 22.476, 21.012, 20.404, 19.389],
    # iz= 9 (world X ≈ 14.2 m)
    [19.304, 17.826, 15.697, 14.228, 14.136, 15.498, 16.948, 18.016, 18.834, 18.583, 18.806, 19.747, 20.529, 20.832, 21.095, 21.999, 21.950, 20.842, 20.872, 20.688],
    # iz=10 (world X ≈ 15.8 m)
    [18.083, 17.339, 15.627, 14.128, 14.239, 15.757, 16.704, 17.121, 17.913, 18.123, 18.435, 19.485, 20.401, 20.552, 20.297, 21.212, 21.528, 21.008, 21.704, 22.197],
    # iz=11 (world X ≈ 17.4 m)
    [16.822, 15.697, 14.692, 14.154, 15.096, 16.565, 16.916, 16.257, 16.536, 17.067, 17.842, 19.074, 19.986, 20.170, 19.552, 20.322, 21.106, 21.463, 22.527, 22.870],
    # iz=12 (world X ≈ 18.9 m)
    [16.804, 14.449, 13.733, 14.353, 15.371, 16.055, 16.230, 15.586, 15.268, 14.955, 15.797, 17.379, 18.308, 18.662, 18.550, 19.490, 20.605, 21.604, 22.790, 23.059],
    # iz=13 (world X ≈ 20.5 m)
    [17.155, 14.594, 13.743, 14.159, 14.428, 14.497, 14.786, 14.791, 14.392, 13.410, 14.003, 15.665, 16.731, 17.432, 18.287, 19.646, 20.433, 20.783, 21.985, 22.822],
    # iz=14 (world X ≈ 22.1 m)
    [17.299, 15.163, 13.628, 13.083, 13.120, 12.919, 12.816, 13.128, 13.153, 12.526, 13.136, 14.644, 15.964, 17.229, 18.389, 19.262, 19.889, 20.244, 20.752, 21.458],
    # iz=15 (world X ≈ 23.7 m)
    [17.846, 15.988, 14.041, 12.924, 12.894, 12.528, 11.924, 12.028, 12.498, 12.716, 13.695, 15.240, 16.738, 18.152, 18.953, 19.258, 19.782, 20.157, 20.055, 20.262],
    # iz=16 (world X ≈ 25.3 m)
    [17.761, 16.114, 14.145, 12.738, 12.391, 11.744, 10.923, 11.145, 12.070, 13.193, 14.398, 15.979, 17.697, 18.888, 19.063, 19.278, 19.565, 19.083, 18.556, 18.804],
    # iz=17 (world X ≈ 26.8 m)
    [17.196, 16.084, 14.352, 12.625, 11.808, 10.723,  9.547,  9.809, 11.085, 12.757, 13.717, 15.053, 16.919, 17.847, 17.763, 18.505, 19.037, 17.951, 16.961, 16.710],
    # iz=18 (world X ≈ 28.4 m)
    [17.509, 16.947, 15.754, 14.070, 12.608, 11.019,  9.479,  9.416, 10.715, 12.026, 12.783, 13.970, 15.421, 16.236, 16.816, 18.038, 18.887, 18.101, 16.775, 15.255],
    # iz=19 (world X ≈ 30.0 m)
    [17.038, 16.356, 15.787, 14.900, 12.961, 11.450, 10.172,  9.439, 10.667, 11.577, 12.769, 14.113, 14.863, 15.745, 16.187, 17.199, 18.188, 17.871, 16.935, 15.055],
], dtype=np.float64)


def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)


class Mavic (Robot):
    # Constants, empirically found.
    K_VERTICAL_THRUST = 68.5  # with this thrust, the drone lifts.
    # Vertical offset where the robot actually targets to stabilize itself.
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0        # P constant of the vertical PID.
    K_ROLL_P = 50.0           # P constant of the roll PID.
    K_PITCH_P = 30.0          # P constant of the pitch PID.

    MAX_YAW_DISTURBANCE = 0.4
    MAX_PITCH_DISTURBANCE = -1
    # Precision between the target position and the robot position in meters
    target_precision = 0.5

    # Obstacle avoidance: minimum altitude above terrain required.
    # Trees can be up to ~10 m tall; 12 m headroom gives a safe buffer above them.
    MIN_SAFE_AGL = 18.0  # metres above terrain (≈ 5 m tree + 7 m clearance)

    def __init__(self):
        Robot.__init__(self)

        self.time_step = int(self.getBasicTimeStep())

        self.water_to_drop = 0

        # Get and enable devices.
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)

        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(1.55)  # vertical PoV
        motors = [self.front_left_motor, self.front_right_motor,
                  self.rear_left_motor, self.rear_right_motor]
        for motor in motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1)

        self.current_pose = 6*[0]  # X,Y,Z, yaw, pitch, roll
        self.target_position = [0, 0, 0]
        self.target_index = 0

        self.world_fire_quadrants = [0, 0]
        self.img_coord_fire = []
        self.WaterDropStatus = False

        # Monoplot-estimated GPS position of the current fire target.
        # Set when fire is detected; cleared after water is dropped.
        self.fire_gps_target = None

        # Registry of all known fire GPS positions discovered during patrol.
        # New entries added by _register_fire(); removed after extinguishing.
        self.known_fires = []

        # Suppresses repeated altitude-bump log lines while the drone is already
        # flying above the minimum safe AGL.
        self._last_altitude_bump = False

    def get_image_from_camera(self):
        """
        Take an image from the camera and prepare it for OpenCV processing:
        - convert data type,
        - convert to RGB format (from BGRA), and
        - rotate & flip to match the actual image.
        Returns:
            image of the camera
        """
        img = self.camera.getImageArray()
        img = np.asarray(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return cv2.flip(img, 1)

    def set_position(self, pos):
        """
        Set a new absolut position of the robot
        Parameters:
            pos (list): [X,Y,Z,yaw,pitch,roll] current absolut position and angles
        """
        self.current_pose = pos

    def move_to_target(self, waypoints, verbose_movement=False, verbose_target=True):
        """
        Move the robot to the given coordinates
        Parameters:
            waypoints (list): list of X,Y coordinates
            verbose_movement (bool): whether to print remaning angle and distance or not
            verbose_target (bool): whether to print targets or not
        Returns:
            yaw_disturbance (float): yaw disturbance (negative value to go on the right)
            pitch_disturbance (float): pitch disturbance (negative value to go forward)
        """

        if self.target_position[0:2] == [0, 0]:  # Initialisation
            self.target_position[0:2] = waypoints[0]
            if verbose_target:
                print("First target: ", self.target_position[0:2])

        # if the robot is at the position with a precision of target_precision
        if all([abs(x1 - x2) < self.target_precision for (x1, x2) in zip(self.target_position, self.current_pose[0:2])]):

            self.target_index += 1
            if self.target_index > len(waypoints)-1:
                self.target_index = 0
            self.target_position[0:2] = waypoints[self.target_index]
            if verbose_target:
                print("Target reached! New target: ",
                      self.target_position[0:2])

        # This will be in ]-pi;pi]
        self.target_position[2] = np.arctan2(
            self.target_position[1] - self.current_pose[1], self.target_position[0] - self.current_pose[0])
        # This is now in ]-2pi;2pi[
        angle_left = self.target_position[2] - self.current_pose[5]
        # Normalize turn angle to ]-pi;pi]
        angle_left = (angle_left + 2*np.pi) % (2*np.pi)
        if (angle_left > np.pi):
            angle_left -= 2*np.pi

        # Turn the robot to the left or to the right according the value and the sign of angle_left
        yaw_disturbance = self.MAX_YAW_DISTURBANCE*angle_left/(2*np.pi)
        # non proportional and decruising function
        pitch_disturbance = clamp(
            np.log10(abs(angle_left)), self.MAX_PITCH_DISTURBANCE, 0.1)

        if verbose_movement:
            distance_left = np.sqrt(((self.target_position[0] - self.current_pose[0]) ** 2) + (
                (self.target_position[1] - self.current_pose[1]) ** 2))
            print("remaning angle: {:.4f}, remaning distance: {:.4f}".format(
                angle_left, distance_left))
        return yaw_disturbance, pitch_disturbance

    def naive_approach(self, verbose=True):
        """
        Naive approach to move the robot above the fire. 
        Closed loop to move the robot towards to the fire step-by-step until it reaches the fire.
        Parameters:
            verbose (bool): whether to print status messages or not
        Returns:
            yaw_disturbance (float): yaw disturbance (negative value to go on the right)
            pitch_disturbance (float): pitch disturbance (negative value to go forward)
        """
        resolutionX, resolutionY = self.camera.getWidth(), self.camera.getHeight()
        x_img, y_img = self.img_coord_fire
        yaw = (self.current_pose[5] + 2*np.pi) % (2*np.pi)
        self.world_fire_quadrants = [0, 0]

        if abs(x_img-resolutionX/2) > 20:
            self.world_fire_quadrants[0] = np.sign(x_img-resolutionX/2)
        if abs(y_img-resolutionY/2) > 20:
            self.world_fire_quadrants[1] = np.sign(y_img-resolutionY/2)
        self.world_fire_quadrants[1] *= np.sign(yaw)
        self.world_fire_quadrants[0] *= -np.sign(yaw)

        yaw_disturbance = self.world_fire_quadrants[0]*clamp(
            abs(x_img-resolutionX/2), 0, self.MAX_YAW_DISTURBANCE)
        pitch_disturbance = self.world_fire_quadrants[1]*clamp(
            abs(y_img-resolutionY/2), 0, abs(self.MAX_PITCH_DISTURBANCE))

        if self.world_fire_quadrants == [0, 0]:
            self.water_to_drop = 15
            if verbose:
                print("Water dropped on fire target: {} at position {}".format(
                    self.target_position[0:2], self.current_pose[0:2]))
            self.img_coord_fire = []

        return yaw_disturbance, pitch_disturbance

    def estimate_fire_gps(self, x_img, y_img, verbose=True):
        """
        Estimate the fire's world GPS coordinates via monoplotting.

        The camera points straight down (pitch motor ~90 deg), so the image
        is a perspective projection of the ground plane.  A pixel displaced
        (nx, ny) from the image centre corresponds to a ground offset:

            off_x = nx * altitude * tan(h_fov / 2)   (along image-x axis)
            off_y = ny * altitude * tan(v_fov / 2)   (along image-y axis)

        Rotating those offsets by the drone's yaw maps them to world-frame
        displacements that are added to the drone's current GPS position.

        Parameters:
            x_img, y_img: pixel coords of the fire in the processed image
            verbose: print the estimate
        Returns:
            [fire_x, fire_y]: estimated world coordinates of the fire
        """
        cam_w = self.camera.getWidth()
        cam_h = self.camera.getHeight()
        h_fov = self.camera.getFov()                              # horizontal FOV [rad]
        v_fov = 2.0 * np.arctan(np.tan(h_fov / 2.0) * cam_h / cam_w)  # vertical FOV

        drone_x  = self.current_pose[0]
        drone_y  = self.current_pose[1]
        yaw      = self.current_pose[5]   # drone heading [rad], 0 = +X world axis
        terrain_z = self.get_terrain_height(drone_x, drone_y)
        altitude = self.current_pose[2] - terrain_z   # true height above ground level [m]

        # Normalised pixel offsets from image centre in [-1, +1]
        nx = (x_img - cam_w / 2.0) / (cam_w / 2.0)
        ny = (y_img - cam_h / 2.0) / (cam_h / 2.0)

        # Ground-plane offsets in the image/camera frame (metres)
        off_x = nx * altitude * np.tan(h_fov / 2.0)
        off_y = ny * altitude * np.tan(v_fov / 2.0)

        # Rotate image-frame offsets into world frame using the 2-D rotation matrix.
        # After ROTATE_90_CLOCKWISE + flip(1), image +x aligns with the drone's
        # forward-right quadrant; the standard rotation matrix handles the full mapping.
        fire_x = drone_x + off_x * np.cos(yaw) - off_y * np.sin(yaw)
        fire_y = drone_y + off_x * np.sin(yaw) + off_y * np.cos(yaw)

        if verbose:
            print("Monoplot: fire at world ({:.1f}, {:.1f})  "
                  "pixel ({:.0f}, {:.0f})  alt {:.1f} m  yaw {:.2f} rad".format(
                  fire_x, fire_y, x_img, y_img, altitude, yaw))
        return [fire_x, fire_y]

    def navigate_to_fire(self, verbose=False):
        """
        Fly directly to the monoplot-estimated fire GPS position.

        Re-uses the same yaw + pitch PID disturbance convention as move_to_target.
        Drops water and clears the target once within the water extinction radius.

        Returns:
            (yaw_disturbance, pitch_disturbance) for motor control
        """
        fire_x, fire_y = self.fire_gps_target

        dist = np.sqrt((fire_x - self.current_pose[0]) ** 2 +
                       (fire_y - self.current_pose[1]) ** 2)

        # Drop water when within 5 m — matches supervisor MAX_EXTINCTION radius
        if dist < 5.0:
            self.water_to_drop = 15
            self._remove_fire(fire_x, fire_y)           # mark this fire as done
            self.fire_gps_target = self._pick_closest_fire()  # re-plan immediately
            print("Water dropped at est. fire ({:.1f}, {:.1f}), "
                  "drone at ({:.1f}, {:.1f}). Next: {}".format(
                  fire_x, fire_y,
                  self.current_pose[0], self.current_pose[1],
                  self.fire_gps_target))
            return 0.0, 0.0

        # Bearing from drone to estimated fire position
        target_yaw = np.arctan2(fire_y - self.current_pose[1],
                                fire_x - self.current_pose[0])
        angle_left = (target_yaw - self.current_pose[5] + 2 * np.pi) % (2 * np.pi)
        if angle_left > np.pi:
            angle_left -= 2 * np.pi

        yaw_disturbance   = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        # +1e-9 prevents log10(0); result is clamped anyway
        pitch_disturbance = clamp(np.log10(abs(angle_left) + 1e-9),
                                  self.MAX_PITCH_DISTURBANCE, 0.1)

        if verbose:
            print("→ fire ({:.1f}, {:.1f})  dist {:.1f} m  angle {:.3f} rad".format(
                  fire_x, fire_y, dist, angle_left))
        return yaw_disturbance, pitch_disturbance

    def fire_detection(self, verbose=True):
        """
        Detect the smoke and return the fire coordinate in the image
        Parameters:
            verbose (bool): whether to print status messages or not
        Returns:
            coord_fire (list):x,y image coordinates of the fire
        """
        img = self.get_image_from_camera()
        # Segment the image by color in HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Range of the smoke
        smoke_lower = np.array([0, 0, 168])
        smoke_upper = np.array([172, 111, 255])

        mask_fire = cv2.inRange(hsv, smoke_lower, smoke_upper)

        fire_ratio = np.round(
            (cv2.countNonZero(mask_fire))/(img.size/3)*100, 2)
        if fire_ratio > 0.15:  # Higher the fire ratio, higher the number of fire in the image

            # Detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
            contours, _ = cv2.findContours(
                image=mask_fire, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

            # Approximate contours to polygons + get circles
            contours_poly = [None]*len(contours)
            centers = [None]*len(contours)
            radius = [None]*len(contours)
            radius_max = 0
            for i, c in enumerate(contours):
                contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                centers[i], radius[i] = cv2.minEnclosingCircle(
                    contours_poly[i])
                # We keep only the biggest circle and > 3
                if radius[i] > 3 and radius[i] > radius_max:
                    coord_fire = centers[i]
                    radius_max = radius[i]
                    if verbose:
                        print(
                            "fire detected, coordinates {}".format(centers[i]))

            if verbose:  # Draw polygonal contour + circles and save the image
                drawing = img.copy()
                for i in range(len(contours)):
                    color = (random.randint(0, 256), random.randint(
                        0, 256), random.randint(0, 256))
                    cv2.drawContours(drawing, contours_poly, i, color)
                    cv2.circle(drawing, (int(centers[i][0]), int(
                        centers[i][1])), int(radius[i]), color, 2)
                cv2.imwrite("fire_detection.jpg", drawing)
            return coord_fire

    def detect_all_fires(self, verbose=True):
        """
        Detect every distinct fire/smoke blob visible in the current camera image.

        Uses the same HSV segmentation as fire_detection() but returns pixel
        coordinates for ALL blobs above the minimum size, not just the largest.
        Uses RETR_EXTERNAL so nested contours of the same blob are not duplicated.

        Returns:
            list of (x_img, y_img) tuples — one per detected fire blob
        """
        img = self.get_image_from_camera()
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        smoke_lower = np.array([0, 0, 168])
        smoke_upper = np.array([172, 111, 255])
        mask_fire = cv2.inRange(hsv, smoke_lower, smoke_upper)

        fire_ratio = np.round((cv2.countNonZero(mask_fire)) / (img.size / 3) * 100, 2)
        if fire_ratio <= 0.15:
            return []

        contours, _ = cv2.findContours(
            mask_fire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        fires = []
        polys, centers_out, radii_out = [], [], []
        for c in contours:
            poly = cv2.approxPolyDP(c, 3, True)
            center, radius = cv2.minEnclosingCircle(poly)
            if radius > 3:
                fires.append(center)
                polys.append(poly)
                centers_out.append(center)
                radii_out.append(radius)

        if verbose and fires:
            drawing = img.copy()
            for poly, center, radius in zip(polys, centers_out, radii_out):
                color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                cv2.drawContours(drawing, [poly], 0, color)
                cv2.circle(drawing, (int(center[0]), int(center[1])), int(radius), color, 2)
            cv2.imwrite("fire_detection.jpg", drawing)
            print("{} fire blob(s) detected in image".format(len(fires)))

        return fires

    def _register_fire(self, gps, merge_radius=5.0):
        """
        Add a fire GPS position to known_fires if not already tracked nearby.
        Capped at 3 fires so drones focus on extinguishing rather than mapping.
        """
        if len(self.known_fires) >= 3:
            return
        for known in self.known_fires:
            if np.linalg.norm(np.array(known) - np.array(gps)) < merge_radius:
                return
        self.known_fires.append(list(gps))
        print("New fire registered at ({:.1f}, {:.1f}), total tracked: {}".format(
            gps[0], gps[1], len(self.known_fires)))

    def _remove_fire(self, fire_x, fire_y, merge_radius=6.0):
        """
        Remove the entry in known_fires closest to (fire_x, fire_y).
        Called after a successful water drop to retire an extinguished fire.
        """
        if not self.known_fires:
            return
        pos = np.array([fire_x, fire_y])
        dists = [np.linalg.norm(np.array(f) - pos) for f in self.known_fires]
        idx = int(np.argmin(dists))
        if dists[idx] < merge_radius:
            removed = self.known_fires.pop(idx)
            print("Fire retired from registry at ({:.1f}, {:.1f}), "
                  "remaining: {}".format(removed[0], removed[1], len(self.known_fires)))

    def _pick_closest_fire(self):
        """
        Return the [x, y] GPS coords of the nearest registered fire, or None.
        Distance is measured from the drone's current GPS position.
        """
        if not self.known_fires:
            return None
        drone_pos = np.array(self.current_pose[:2])
        dists = [np.linalg.norm(np.array(f) - drone_pos) for f in self.known_fires]
        return list(self.known_fires[int(np.argmin(dists))])

    def get_terrain_height(self, world_x, world_y):
        """
        Bilinear interpolation of terrain height (world-Z) at (world_x, world_y).

        The ElevationGrid's 120° rotation gives:
            iz = world_x / spacing  →  selects the row  (local Z axis)
            ix = world_y / spacing  →  selects the col  (local X axis)
        Returns the interpolated terrain world-Z in metres.
        """
        s = _TERRAIN_SPACING
        fx = np.clip(world_y / s, 0.0, _TERRAIN_X_DIM - 1.0)  # fractional ix
        fz = np.clip(world_x / s, 0.0, _TERRAIN_Z_DIM - 1.0)  # fractional iz

        x0, z0 = int(fx), int(fz)
        x1 = min(x0 + 1, _TERRAIN_X_DIM - 1)
        z1 = min(z0 + 1, _TERRAIN_Z_DIM - 1)
        dx, dz = fx - x0, fz - z0

        return float(
            _TERRAIN_H[z0, x0] * (1 - dx) * (1 - dz) +
            _TERRAIN_H[z0, x1] *      dx  * (1 - dz) +
            _TERRAIN_H[z1, x0] * (1 - dx) *      dz  +
            _TERRAIN_H[z1, x1] *      dx  *       dz
        )

    def run(self):
        t1 = self.getTime()
        t2 = self.getTime()
        t3 = self.getTime()

        roll_disturbance = 0
        pitch_disturbance = 0
        yaw_disturbance = 0

        # We add controller args to waypoints and target_altitude variables
        opt_parser = optparse.OptionParser()
        opt_parser.add_option("--patrol_coords", default="11 11, 11 21, 21 21,21 11",
                              help="Specify the patrol coordinates in the format [x1 y1, x2 y2, ...]")
        opt_parser.add_option("--target_altitude", default=42,
                              type=float, help="target altitude of the robot in meters")
        options, _ = opt_parser.parse_args()

        point_list = options.patrol_coords.split(',')
        number_of_waypoints = len(point_list)
        waypoints = []
        for i in range(0, number_of_waypoints):
            waypoints.append([])
            waypoints[i].append(float(point_list[i].split()[0]))
            waypoints[i].append(float(point_list[i].split()[1]))

        target_altitude = options.target_altitude

        while self.step(self.time_step) != -1:

            # Read sensors
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            Xpos, Ypos, altitude = self.gps.getValues()
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
            self.set_position([Xpos, Ypos, altitude, roll, pitch, yaw])

            # Drop the water from the drone
            if self.water_to_drop > 0:
                self.WaterDropStatus = True
                self.setCustomData(str(self.water_to_drop))
                self.water_to_drop = 0
            else:
                self.setCustomData(str(0))

            if altitude > target_altitude - 1:
                # Motion
                if self.getTime() - t1 > 0.1:
                    if self.fire_gps_target is not None:
                        # Monoplot navigation: fly directly to estimated fire coords
                        yaw_disturbance, pitch_disturbance = self.navigate_to_fire()
                    else:
                        yaw_disturbance, pitch_disturbance = self.move_to_target(
                            waypoints)
                    t1 = self.getTime()
                # Fire detection: register all visible blobs; re-plan to closest
                if self.getTime() - t2 > 1:
                    if not self.WaterDropStatus:
                        all_coords = self.detect_all_fires()
                        for coord in all_coords:
                            gps = self.estimate_fire_gps(*coord, verbose=False)
                            self._register_fire(gps)
                        # Re-evaluate every cycle — switch target if a closer fire exists
                        new_target = self._pick_closest_fire()
                        if new_target != self.fire_gps_target:
                            if new_target is not None:
                                print("Re-planning → fire at ({:.1f}, {:.1f}), "
                                      "{} fire(s) tracked".format(
                                      new_target[0], new_target[1],
                                      len(self.known_fires)))
                            self.fire_gps_target = new_target
                    t2 = self.getTime()

                if not self.WaterDropStatus:
                    t3 = self.getTime()
                if self.getTime() - t3 > 15:  # Wait 15 times to avoid detection of the dropping water as smoke
                    self.WaterDropStatus = False

            roll_input = self.K_ROLL_P * \
                clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
            pitch_input = self.K_PITCH_P * \
                clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            yaw_input = yaw_disturbance

            # Obstacle avoidance: compute safe altitude from terrain + look-ahead.
            # Check both current position and 8 m ahead in the drone's heading
            # direction so the altitude is raised *before* reaching the obstacle.
            terrain_now  = self.get_terrain_height(Xpos, Ypos)
            terrain_next = self.get_terrain_height(
                Xpos + 8.0 * np.cos(yaw),
                Ypos + 8.0 * np.sin(yaw))
            terrain_h        = max(terrain_now, terrain_next)
            effective_target = max(target_altitude, terrain_h + self.MIN_SAFE_AGL)

            if effective_target > target_altitude:
                if not self._last_altitude_bump:
                    print("Obstacle avoidance: terrain {:.1f} m ahead → "
                          "altitude target raised from {:.0f} to {:.1f} m".format(
                          terrain_h, target_altitude, effective_target))
                self._last_altitude_bump = True
            else:
                self._last_altitude_bump = False

            clamped_difference_altitude = clamp(
                effective_target - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            vertical_input = self.K_VERTICAL_P * \
                pow(clamped_difference_altitude, 3.0)

            front_left_motor_input = self.K_VERTICAL_THRUST + \
                vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = self.K_VERTICAL_THRUST + \
                vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input = self.K_VERTICAL_THRUST + \
                vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = self.K_VERTICAL_THRUST + \
                vertical_input - yaw_input - pitch_input + roll_input

            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)


robot = Mavic()
robot.run()
