from datas.identity3_B_lora_data.Config import DATA_FILE, DATA_LABEL
from tools.dataloader import DataLoader
from trajectories.gps import GPSTrajectories # GPS 데이터
from trajectories.EKF import EKFTrajectories # EKF
from trajectories.EKF_nogps import EKFTrajectories2
from tools.plot import *

data = DataLoader(DATA_FILE, DATA_LABEL)
gps = GPSTrajectories(data.gps_latitude, data.gps_longitude, data.altitude, data.time)
ekf = EKFTrajectories(data.acc, data.gyro, data.mag, data.gps_latitude, data.gps_longitude, data.altitude, data.time, data.launch_idx)
ekf2 = EKFTrajectories2(data.acc, data.gyro, data.mag, data.altitude, data.time, data.launch_idx)
plot_multiple([gps, ekf, ekf2], ["GPS", "EKF", "EKF2"], ["r", "g", "b"])