import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

def set_equal_aspect_3d(ax, x, y, z):
    extents = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()])
    max_extent = extents.max()
    centers = np.array([(x.max()+x.min())/2, (y.max()+y.min())/2, (z.max()+z.min())/2])
    ax.set_xlim(centers[0] - max_extent/2, centers[0] + max_extent/2)
    ax.set_ylim(centers[1] - max_extent/2, centers[1] + max_extent/2)
    ax.set_zlim(centers[2] - max_extent/2, centers[2] + max_extent/2)
    ax.set_box_aspect([1,1,1])

def plot(trajectories, label):
    x, y, z = trajectories.x, trajectories.y, trajectories.z
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, 'b-', label=label)
    set_equal_aspect_3d(ax, x, y, z)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('Trajectory Plot')
    ax.legend()
    plt.show()


def save_trajectory_video(trajectories, filename='test', fps=20):
    x, y, z, t = trajectories.x, trajectories.y, trajectories.z, trajectories.time
    dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    dt = np.diff(t)
    speeds = np.concatenate(([0], dist / dt))
    max_speeds = np.maximum.accumulate(speeds)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], 'b-', lw=2)
    point, = ax.plot([], [], [], 'ro')

    stats_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12,
                           bbox=dict(facecolor='white', alpha=0.7))

    n_points = len(x)
    pause_frames = fps
    total_frames = n_points + pause_frames

    def init():
        set_equal_aspect_3d(ax, x, y, z)
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_zlabel('Altitude (m)')
        stats_text.set_text("")
        return line, point, stats_text

    def update(i):
        idx = min(i, n_points - 1)
        line.set_data(x[:idx + 1], y[:idx + 1])
        line.set_3d_properties(z[:idx + 1])
        point.set_data([x[idx]], [y[idx]])
        point.set_3d_properties([z[idx]])

        content = (f"Time: {t[idx]:.2f} s\n"
                   f"Altitude: {z[idx]:.1f} m\n"
                   f"Speed: {speeds[idx]:.2f} m/s\n"
                   f"Max Speed: {max_speeds[idx]:.2f} m/s")
        stats_text.set_text(content)

        return line, point, stats_text

    ani = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=False)
    pbar = tqdm(total=total_frames, desc="영상 생성 중")

    try:
        writer = FFMpegWriter(fps=fps)
        ani.save(f"{filename}.mp4", writer=writer, progress_callback=lambda i, n: pbar.update(1))
    except Exception:
        pbar.close()
        pbar = tqdm(total=total_frames, desc="GIF 생성 중")
        ani.save(f'{filename}.gif', writer='pillow', fps=fps, progress_callback=lambda i, n: pbar.update(1))
    finally:
        pbar.close()

def plot_multiple(trajectories, labels, colors):
    xs = [i.x for i in trajectories]
    ys = [i.y for i in trajectories]
    zs = [i.z for i in trajectories]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    all_z = np.concatenate(zs)

    for x, y, z, label, color in zip(xs, ys, zs, labels, colors):
        ax.plot(x, y, z, color=color, linestyle='-', label=label)

    set_equal_aspect_3d(ax, all_x, all_y, all_z)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('Trajectory Plot')
    ax.legend()
    plt.show()


def save_trajectory_video_multiple(trajectories, labels, colors, filename='test', fps=20):
    xs = [i.x for i in trajectories]
    ys = [i.y for i in trajectories]
    zs = [i.z for i in trajectories]
    t = trajectories[0].time

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    lines = []
    points = []

    for color, label in zip(colors, labels):
        line, = ax.plot([], [], [], color=color, linestyle='-', lw=2, label=label)
        point, = ax.plot([], [], [], color=color, marker='o')
        lines.append(line)
        points.append(point)

    n_points = len(t)
    pause_frames = fps
    total_frames = n_points + pause_frames

    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    all_z = np.concatenate(zs)

    ax.legend()

    def init():
        set_equal_aspect_3d(ax, all_x, all_y, all_z)
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_zlabel('Altitude (m)')
        return lines + points

    def update(i):
        idx = min(i, n_points - 1)
        for j in range(len(xs)):
            lines[j].set_data(xs[j][:idx + 1], ys[j][:idx + 1])
            lines[j].set_3d_properties(zs[j][:idx + 1])
            points[j].set_data([xs[j][idx]], [ys[j][idx]])
            points[j].set_3d_properties([zs[j][idx]])
        return lines + points

    ani = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=False)
    pbar = tqdm(total=total_frames, desc="영상 생성 중")

    try:
        writer = FFMpegWriter(fps=fps)
        ani.save(f"{filename}.mp4", writer=writer, progress_callback=lambda i, n: pbar.update(1))
    except Exception:
        pbar.close()
        pbar = tqdm(total=total_frames, desc="GIF 생성 중")
        ani.save(f'{filename}.gif', writer='pillow', fps=fps, progress_callback=lambda i, n: pbar.update(1))
    finally:
        pbar.close()