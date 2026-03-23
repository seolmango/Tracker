import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import block_diag


class EKFTrajectories:
    def __init__(self, acc_data, gyro_data, mag_data, lat_data, lon_data, alt_data, time_data, start_idx):
        self.acc_raw = acc_data
        self.gyro_raw = gyro_data
        self.mag_raw = mag_data
        self.lat = lat_data
        self.lon = lon_data
        self.alt = alt_data
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

        self._latlon_to_enu()
        self._calibrate_sensors()
        self._compute()

    def _latlon_to_enu(self):
        lats = np.array(self.lat) / 1e7
        lons = np.array(self.lon) / 1e7
        alts = np.array(self.alt)

        phi = np.radians(lats)
        lam = np.radians(lons)

        a = 6378137.0
        f = 1 / 298.257223563
        e2 = f * (2 - f)

        n = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
        x_ecef = (n + alts) * np.cos(phi) * np.cos(lam)
        y_ecef = (n + alts) * np.cos(phi) * np.sin(lam)
        z_ecef = (n * (1 - e2) + alts) * np.sin(phi)

        x0, y0, z0 = x_ecef[0], y_ecef[0], z_ecef[0]
        phi0, lam0 = phi[0], lam[0]

        dx = x_ecef - x0
        dy = y_ecef - y0
        dz = z_ecef - z0

        self.gps_x = -np.sin(lam0) * dx + np.cos(lam0) * dy
        self.gps_y = -np.sin(phi0) * np.cos(lam0) * dx - np.sin(phi0) * np.sin(lam0) * dy + np.cos(phi0) * dz
        self.gps_z = np.cos(phi0) * np.cos(lam0) * dx + np.cos(phi0) * np.sin(lam0) * dy + np.sin(phi0) * dz

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
        self.p = np.array([self.gps_x[self.start_idx], self.gps_y[self.start_idx], self.gps_z[self.start_idx]])
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

        last_gps_meas = np.array([self.gps_x[self.start_idx], self.gps_y[self.start_idx], self.gps_z[self.start_idx]])

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

            curr_gps = np.array([self.gps_x[i], self.gps_y[i], self.gps_z[i]])
            if np.linalg.norm(curr_gps - last_gps_meas) > 1e-6:
                y_list.append(curr_gps - self.p)
                H_gps = np.zeros((3, 15))
                H_gps[0:3, 0:3] = np.eye(3)
                H_list.append(H_gps)
                R_list.append(np.eye(3) * 2.0)
                last_gps_meas = curr_gps

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

        self.x[:self.start_idx + 1] = self.gps_x[self.start_idx]
        self.y[:self.start_idx + 1] = self.gps_y[self.start_idx]
        self.z[:self.start_idx + 1] = self.gps_z[self.start_idx]