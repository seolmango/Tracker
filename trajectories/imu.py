import numpy as np
from scipy.spatial.transform import Rotation as R


class IMUTrajectories:
    def __init__(self, acc_data, gyro_data, mag_data, time_data, start_idx):
        self.acc_raw = acc_data
        self.gyro_raw = gyro_data
        self.mag_raw = mag_data
        self.time = time_data / 1000.0
        self.start_idx = start_idx

        n_samples = len(self.time)
        self.x = np.zeros(n_samples)
        self.y = np.zeros(n_samples)
        self.z = np.zeros(n_samples)

        self._calibrate_sensors()
        self._compute()

    def _calibrate_sensors(self):
        calib_range_acc = self.acc_raw[:self.start_idx]
        calib_range_gyro = self.gyro_raw[:self.start_idx]
        calib_range_mag = self.mag_raw[:self.start_idx]

        if len(calib_range_acc) > 0:
            self.gyro_bias = np.mean(calib_range_gyro, axis=0)
            self.init_acc_mean = np.mean(calib_range_acc, axis=0)
            self.init_mag_mean = np.mean(calib_range_mag, axis=0)
            self.g_magnitude = np.linalg.norm(self.init_acc_mean)

            print(f"[IMU] gyro bias: {self.gyro_bias}, g_magnitude: {self.g_magnitude:.3f}")
        else:
            self.gyro_bias = np.zeros(3)
            self.init_acc_mean = np.array([0.0, 0.0, 9.80665])
            self.init_mag_mean = np.array([1.0, 0.0, 0.0])
            self.g_magnitude = 9.80665

        self.gyro = np.radians(self.gyro_raw - self.gyro_bias)

    def _get_initial_rotation(self):
        up_vec = self.init_acc_mean / np.linalg.norm(self.init_acc_mean)
        mag_norm = self.init_mag_mean / np.linalg.norm(self.init_mag_mean)

        east_vec = np.cross(mag_norm, up_vec)
        east_vec = east_vec / np.linalg.norm(east_vec)

        north_vec = np.cross(up_vec, east_vec)
        north_vec = north_vec / np.linalg.norm(north_vec)

        rot_matrix = np.vstack((east_vec, north_vec, up_vec)).T
        return R.from_matrix(rot_matrix)

    def _compute(self):
        n_samples = len(self.time)
        rot_init = self._get_initial_rotation()

        curr_rot = rot_init
        curr_vel = np.array([0.0, 0.0, 0.0])
        curr_pos = np.array([0.0, 0.0, 0.0])

        g_world = np.array([0.0, 0.0, self.g_magnitude])
        prev_acc_world = curr_rot.apply(self.acc_raw[self.start_idx]) - g_world

        for i in range(self.start_idx + 1, n_samples):
            dt = self.time[i] - self.time[i - 1]
            if dt <= 0: continue

            mid_gyro = (self.gyro[i] + self.gyro[i - 1]) / 2.0
            delta_rot = R.from_rotvec(mid_gyro * dt)
            curr_rot = curr_rot * delta_rot

            acc_body = self.acc_raw[i]
            acc_world = curr_rot.apply(acc_body) - g_world

            new_vel = curr_vel + (prev_acc_world + acc_world) / 2.0 * dt
            new_pos = curr_pos + (curr_vel + new_vel) / 2.0 * dt

            curr_vel = new_vel
            curr_pos = new_pos
            prev_acc_world = acc_world

            self.x[i], self.y[i], self.z[i] = curr_pos