import numpy as np

class GPSTrajectories:
    def __init__(self, lat_data, lon_data, alt_rel, time):
        lats = np.array(lat_data) / 1e7
        lons = np.array(lon_data) / 1e7
        alts = np.array(alt_rel)

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

        self.x = -np.sin(lam0) * dx + np.cos(lam0) * dy
        self.y = -np.sin(phi0) * np.cos(lam0) * dx - np.sin(phi0) * np.sin(lam0) * dy + np.cos(phi0) * dz
        self.z = np.cos(phi0) * np.cos(lam0) * dx + np.cos(phi0) * np.sin(lam0) * dy + np.sin(phi0) * dz
        self.time = np.array(time) / 1000