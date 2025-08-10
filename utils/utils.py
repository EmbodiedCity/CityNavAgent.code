import os
import cv2
import airsim
import torch
import textwrap

import torch.distributed as dist

import math
import numpy as np

from src.common.param import args
from maps import compute_shortest_path


def to_eularian_angles(xyzw):
    # four eles in xyzw

    z = xyzw[2]
    y = xyzw[1]
    x = xyzw[0]
    w = xyzw[3]
    ysqr = y * y

    # roll (x-axis rotation)
    t0 = +2.0 * (w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + ysqr)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w*y - z*x)
    if (t2 > 1.0):
        t2 = 1
    if (t2 < -1.0):
        t2 = -1.0
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w*z + x*y)
    t4 = +1.0 - 2.0 * (ysqr + z*z)
    yaw = math.atan2(t3, t4)

    return np.array([pitch, roll, yaw])


def compute_airsim_yaw(x, y):
    # x, y in airsim coords, since airsim (x, y) is equal to (y, x) in standard coordinate system
    # np.arctan2(y, x) is in standard coords, and in [-pi, pi]
    # thus, first convert airsim (x,y) in standard coords (y, x), and use np.arctan2
    # since yaw is relative angle to standard y coords, thus use pi/2 - np.arctan2
    std_x = y
    std_y = x
    std_rot = np.arctan2(std_y, std_x)
    yaw = np.pi / 2 - std_rot   # yaw in [-pi/2, 3*pi/2]
    if np.pi <= yaw <= 3*np.pi/2:
        yaw = yaw - 2*np.pi     # convert yaw in [-pi, pi]

    return yaw


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.DistributedDataParallel = False
        return

    args.DistributedDataParallel = True

    torch.cuda.set_device(gpu)
    print('distributed init (rank {}, word {})'.format(rank, world_size), flush=True)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    torch.distributed.barrier()


def manual_init_distributed_mode(rank, world_size, local_rank):
    args.DistributedDataParallel = True
    args.batchSize = 1

    gpu = local_rank
    torch.cuda.set_device(gpu)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.DDP_MASTER_PORT)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)

    print('distributed init (rank {}, word {})'.format(rank, world_size), flush=True)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    torch.distributed.barrier()


def FromPortGetPid(port: int):
    import subprocess
    import time
    import signal

    subprocess_execute = "netstat -nlp | grep {}".format(
        port,
    )

    try:
        p = subprocess.Popen(
            subprocess_execute,
            stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True,
        )
    except Exception as e:
        print(
            "{}\t{}\t{}".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                'FromPortGetPid',
                e,
            )
        )
        return None
    except:
        return None

    pid = None
    for line in iter(p.stdout.readline, b''):
        line = str(line, encoding="utf-8")
        if 'tcp' in line:
            pid = line.strip().split()[-1].split('/')[0]
            try:
                pid = int(pid)
            except:
                pid = None
            break

    try:
        # os.system(("kill -9 {}".format(p.pid)))
        os.kill(p.pid, signal.SIGKILL)
    except:
        pass

    return pid


def non_maximum_suppression_1d(signal, window_size):
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    half_window = window_size // 2
    length = len(signal)
    result = np.zeros(length)

    for i in range(length):
        start = max(0, i - half_window)
        end = min(length, i + half_window + 1)
        if signal[i] == max(signal[start:end]):
            result[i] = signal[i]

    return result


def calculate_movement_steps(A, B, h_ss=5.0, v_ss=2.0, yaw_ss=15):
    if isinstance(A, airsim.Pose):
        # A, B are airsim pose
        A_pos = np.array(list(A.position))
        A_ori = airsim.to_eularian_angles(A.orientation)  # p, r, y
    elif isinstance(A, list) or isinstance(A, np.ndarray):
        A_pos = np.array(A[:3])

        if len(A) == 6:
            A_ori = np.array(A[3:6])
        elif len(A) > 6:
            A_ori = to_eularian_angles(A[3:7])
        else:
            raise ValueError(f"invalid orientation: {A}")
    else:
        raise ValueError(f"invalid pose format: {A}")

    if isinstance(B, airsim.Pose):
        # A, B are airsim pose
        B_pos = np.array(list(B.position))
        B_ori = airsim.to_eularian_angles(B.orientation)  # p, r, y
    elif isinstance(B, list) or isinstance(B, np.ndarray):
        B_pos = np.array(B[:3])

        if len(B) == 6:
            B_ori = np.array(B[3:])
        elif len(B) > 6:
            B_ori = to_eularian_angles(B[3:7])
        elif len(B) == 3:
            del_pos = B_pos - A_pos
            B_yaw = compute_airsim_yaw(del_pos[0], del_pos[1])
            B_ori = [0, 0, B_yaw]
        else:
            raise ValueError(f"invalid orientation: {B}")
    else:
        raise ValueError(f"invalid pose format: {B}")

    delta_yaw = B_ori[-1] - A_ori[-1]
    abs_delta_yaw = np.abs(delta_yaw)
    rot_angle = abs_delta_yaw if abs_delta_yaw < np.pi else 2*np.pi-abs_delta_yaw
    rot_reverse = 1 if abs_delta_yaw < np.pi else -1
    rot_direction = np.sign(delta_yaw) * rot_reverse

    delta_pos_h = B_pos[:2] - A_pos[:2]
    delta_pos_v = B_pos[-1] - A_pos[-1]

    h_dist = np.linalg.norm(delta_pos_h)
    v_dist = np.abs(delta_pos_v)
    h_ori = delta_pos_h / (h_dist+1e-6)
    h_step_size = int(h_dist // h_ss)
    v_step_size = int(v_dist // v_ss)
    x_step, y_step = h_ori * h_ss
    x_dist, y_dist, z_dist = np.abs(delta_pos_h[0]), np.abs(delta_pos_h[1]), v_dist
    x_dire, y_dire, z_dire = np.sign(delta_pos_h[0]), np.sign(delta_pos_h[1]), np.sign(delta_pos_v)

    # delta_yaw_inter = np.append(np.arange(0, np.abs(delta_yaw), np.deg2rad(yaw_ss)) * np.sign(delta_yaw), delta_yaw)
    delta_yaw_inter = np.append(np.arange(0, rot_angle, np.deg2rad(yaw_ss)) * rot_direction, rot_angle * rot_direction)

    delta_pos_x_inter = np.array([0 + x_step * (i + 1) for i in range(h_step_size)] + [x_dist]) * x_dire if h_dist > 0 else []
    delta_pos_y_inter = np.array([0 + y_step * (i + 1) for i in range(h_step_size)] + [y_dist]) * y_dire if h_dist > 0 else []
    delta_pos_z_inter = np.array([0 + v_ss * (i + 1) for i in range(v_step_size)] + [z_dist]) * z_dire if v_dist > 0 else []

    yaw_inter = A_ori[-1] + delta_yaw_inter
    yaw_inter = (yaw_inter + np.pi) % (2 * np.pi) - np.pi
    pos_x_inter = A_pos[0] + delta_pos_x_inter
    pos_y_inter = A_pos[1] + delta_pos_y_inter
    pos_z_inter = A_pos[2] + delta_pos_z_inter

    path = []
    for yaw_it in yaw_inter:
        path.append([A_pos[0], A_pos[1], A_pos[2], A_ori[0], A_ori[1], yaw_it])

    for pos_x_it, pos_y_it in zip(pos_x_inter, pos_y_inter):
        path.append([pos_x_it, pos_y_it, path[-1][2], path[-1][3], path[-1][4], path[-1][5]])

    for pos_z_it in pos_z_inter:
        path.append([path[-1][0], path[-1][1], pos_z_it, path[-1][3], path[-1][4], path[-1][5]])

    # for p in path:
    #     print(p)
    # print(len(path))

    return len(path), path


def calculate_movement_steps_mem(mem_g, mem_traj, h_ss=5.0, v_ss=2.0, y_ss=15):
    step_size = 0.0
    action_traj = []
    for ii in range(1, len(mem_traj)):
        mid_path, mid_nodes = compute_shortest_path(mem_g, mem_traj[ii - 1], mem_traj[ii])
        for jj in range(1, len(mid_path)):
            start_coord = mid_path[jj-1][:3]
            tar_coord = mid_path[jj][:3]
            sz, mid_coords = calculate_movement_steps(start_coord, tar_coord, h_ss, v_ss, y_ss)
            action_traj.extend(mid_coords)
            step_size += sz

    return step_size, action_traj


def append_text_to_image(image: np.ndarray, text: str):
    r"""Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final


if __name__ == "__main__":
    calculate_movement_steps_v2(0, 0)
