import pandas as pd
import numpy as np

class DataLoader:
    """
    로켓 데이터 csv 데이터를 로딩하는 클래스입니다.
    """
    def __init__(self, file_path, data_label=None):
        if data_label is None:
            data_label = {}
        self.data_label = data_label
        self.file_path = file_path
        self.standby_idx = 0
        self.launch_idx = 0

        self._load_csv()
        self._find_phases()
        self._set_initial_state()

    def _load_csv(self):
        df = pd.read_csv(self.file_path)

        self.acc = np.column_stack([
            df[self.data_label.get("Acc_x", "Acc_x")].to_numpy(),
            df[self.data_label.get("Acc_y", "Acc_y")].to_numpy(),
            df[self.data_label.get("Acc_z", "Acc_z")].to_numpy()
        ])

        self.gyro = np.column_stack([
            df[self.data_label.get("Gyro_x", "Gyro_x")].to_numpy(),
            df[self.data_label.get("Gyro_y", "Gyro_y")].to_numpy(),
            df[self.data_label.get("Gyro_z", "Gyro_z")].to_numpy()
        ])

        self.mag = np.column_stack([
            df[self.data_label.get("Mag_x", "Mag_x")].to_numpy(),
            df[self.data_label.get("Mag_y", "Mag_y")].to_numpy(),
            df[self.data_label.get("Mag_z", "Mag_z")].to_numpy()
        ])

        self.gps_altitude = df[self.data_label.get("Gps_alt", "Gps_alt")].to_numpy()
        self.gps_latitude = df[self.data_label.get("Gps_lat", "Gps_lat")].to_numpy()
        self.gps_longitude = df[self.data_label.get("Gps_lon", "Gps_lon")].to_numpy()

        self.pressure = df[self.data_label.get("Pressure", "Pressure")].to_numpy()
        self.temperature = df[self.data_label.get("Temperature", "Temperature")].to_numpy()
        self.time = df[self.data_label.get("Time", "Time")].to_numpy()

        self.length = len(self.acc)

    def _find_phases(self):
        g0_size = np.linalg.norm(self.acc[self.standby_idx])
        acc_size = np.linalg.norm(self.acc, axis=1)

        launch_idx = 0
        while launch_idx < self.length and abs(g0_size - acc_size[launch_idx]) < 0.05 * g0_size:
            launch_idx += 1
        self.launch_idx = launch_idx

    def _set_initial_state(self):
        ground_pressure = np.mean(self.pressure[:self.launch_idx])
        ground_temperature = np.mean(self.temperature[:self.launch_idx])
        self.altitude = ((ground_pressure / self.pressure) ** (1/5.257) - 1) * (ground_temperature + 273.15) / 0.0065