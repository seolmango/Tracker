import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import block_diag

class EKFTrajectories2:
    def __init__(self, acc_data, gyro_data, mag_data, alt_data, time_data, start_idx):
        self.acc_raw = acc_data
        self.gyro_raw = gyro_data
        self.mag_raw = mag_data
        self.alt_data = alt_data
        self.time = time_data / 1000.0
        self.start_idx = start_idx

        n_samples = len(self.time)
        self.x = np.zeros(n_samples)
        self.y = np.zeros(n_samples)
        self.z = np.zeros(n_samples)

        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = R.identity()
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)

        self._calibrate_sensors()
        self._compute()

    def _calibrate_sensors(self):
        calib_acc = self.acc_raw[:self.start_idx]
        calib_gyro = self.gyro_raw[:self.start_idx]
        calib_mag = self.mag_raw[:self.start_idx]

        if len(calib_acc) > 0:
            self.bg = np.mean(calib_gyro, axis=0)
            self.init_acc_mean = np.mean(calib_acc, axis=0)
            self.init_mag_mean = np.mean(calib_mag, axis=0)
            self.g_magnitude = np.linalg.norm(self.init_acc_mean)
        else:
            self.bg = np.zeros(3)
            self.init_acc_mean = np.array([0.0, 0.0, 9.80665])
            self.init_mag_mean = np.array([1.0, 0.0, 0.0])
            self.g_magnitude = 9.80665

        self.gyro_rad = np.radians(self.gyro_raw)
        self.bg = np.radians(self.bg)

    def _get_initial_rotation(self):
        up_vec = self.init_acc_mean / np.linalg.norm(self.init_acc_mean)
        mag_norm = self.init_mag_mean / np.linalg.norm(self.init_mag_mean)

        east_vec = np.cross(mag_norm, up_vec)
        east_vec = east_vec / np.linalg.norm(east_vec)

        north_vec = np.cross(up_vec, east_vec)
        north_vec = north_vec / np.linalg.norm(north_vec)

        rot_matrix = np.vstack((east_vec, north_vec, up_vec)).T
        return R.from_matrix(rot_matrix)

    def _skew(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def _compute(self):
        n_samples = len(self.time)

        self.q = self._get_initial_rotation()
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.ba = np.zeros(3)

        g_vec = np.array([0.0, 0.0, -self.g_magnitude])

        mag_norm_init = self.init_mag_mean / np.linalg.norm(self.init_mag_mean)
        m_global = self.q.apply(mag_norm_init)

        P = np.eye(15) * 0.01

        Q = np.eye(15)
        Q[0:3, 0:3] *= 1e-4
        Q[3:6, 3:6] *= 1e-3
        Q[6:9, 6:9] *= 1e-4
        Q[9:12, 9:12] *= 1e-5
        Q[12:15, 12:15] *= 1e-5

        for i in range(self.start_idx + 1, n_samples):
            dt = self.time[i] - self.time[i - 1]
            if dt <= 0: continue

            omega = self.gyro_rad[i] - self.bg
            acc_body = self.acc_raw[i] - self.ba

            delta_q = R.from_rotvec(omega * dt)
            self.q = self.q * delta_q

            C = self.q.as_matrix()
            acc_world = C @ acc_body + g_vec

            self.p = self.p + self.v * dt + 0.5 * acc_world * dt ** 2
            self.v = self.v + acc_world * dt

            F = np.eye(15)
            F[0:3, 3:6] = np.eye(3) * dt
            F[3:6, 6:9] = -C @ self._skew(acc_body) * dt
            F[3:6, 9:12] = -C * dt
            F[6:9, 6:9] = np.eye(3) - self._skew(omega) * dt
            F[6:9, 12:15] = -np.eye(3) * dt

            P = F @ P @ F.T + Q * dt

            y_list = []
            H_list = []
            R_list = []

            mag_meas = self.mag_raw[i] / np.linalg.norm(self.mag_raw[i])
            m_expected = self.q.inv().apply(m_global)

            y_list.append(mag_meas - m_expected)
            H_mag = np.zeros((3, 15))
            H_mag[:, 6:9] = self._skew(m_expected)
            H_list.append(H_mag)
            R_list.append(np.eye(3) * 0.5)

            baro_meas = self.alt_data[i]
            y_list.append(np.array([baro_meas - self.p[2]]))
            H_baro = np.zeros((1, 15))
            H_baro[0, 2] = 1.0
            H_list.append(H_baro)
            R_list.append(np.array([[1.0]]))

            if np.linalg.norm(acc_world) < 0.2:
                y_list.append(np.zeros(3) - self.v)
                H_zupt = np.zeros((3, 15))
                H_zupt[0:3, 3:6] = np.eye(3)
                H_list.append(H_zupt)
                R_list.append(np.eye(3) * 1e-4)

            y = np.concatenate(y_list)
            H = np.vstack(H_list)
            R_cov = block_diag(*R_list)

            S = H @ P @ H.T + R_cov
            K = P @ H.T @ np.linalg.inv(S)
            dx = K @ y

            self.p += dx[0:3]
            self.v += dx[3:6]

            dq = R.from_rotvec(dx[6:9])
            self.q = self.q * dq

            self.ba += dx[9:12]
            self.bg += dx[12:15]

            P = (np.eye(15) - K @ H) @ P

            self.x[i] = self.p[0]
            self.y[i] = self.p[1]
            self.z[i] = self.p[2]

        self.x[:self.start_idx + 1] = 0.0
        self.y[:self.start_idx + 1] = 0.0
        self.z[:self.start_idx + 1] = 0.0