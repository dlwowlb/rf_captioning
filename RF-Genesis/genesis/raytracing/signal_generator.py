from tqdm import tqdm

import torch
import numpy as np
from .radar import Radar
from PIL import Image
import math
torch.set_default_device('cuda')

def calculate_environment_points(environment_pir):
    """
    environment_pir: (H, W, 3) torch tensor, assumed to be on the correct device (e.g., CUDA)
    Returns: (H*W, 3) point cloud tensor in Mitsuba camera space.

    Mitsuba camera convention:
      x = right, y = up, z = -forward (camera looks along -z)
    Image convention:
      i = column (horizontal, left-to-right), j = row (vertical, top-to-bottom)
    """
    H, W, _ = environment_pir.shape
    device = environment_pir.device

    distance = environment_pir[:, :, 0] * 5 + 5  # [H, W]

    fov_rad = math.radians(60)
    fx = W / (2 * math.tan(fov_rad / 2))
    fy = fx
    cx = W / 2
    cy = H / 2

    j = torch.arange(0, H, device=device).view(-1, 1).expand(H, W)  # rows
    i = torch.arange(0, W, device=device).view(1, -1).expand(H, W)  # cols

    # Mitsuba camera frame: x=right, y=up (flip image y), z=-forward (negate)
    x = (i - cx) / fx
    y = -(j - cy) / fy   # flip: image y-down -> camera y-up
    z = -torch.ones_like(x, device=device)  # camera looks along -z

    xyz = torch.stack((x, y, z), dim=-1) * distance.unsqueeze(-1)  # [H, W, 3]
    points = xyz.reshape(-1, 3)  # [H*W, 3]
    return points


def camera_to_world_points(points, sensor_origin, sensor_target):
    """
    카메라 좌표계의 pointcloud를 세계 좌표계로 변환.
    Mitsuba의 look_at convention에 맞춰 변환 행렬을 구성한다.
    
    Args:
        points: (N, 3) 카메라 좌표계의 점들
        sensor_origin: [x, y, z] 세계좌표에서의 센서 위치
        sensor_target: [x, y, z] 세계좌표에서의 센서가 바라보는 지점
    Returns:
        (N, 3) 세계 좌표계의 점들
    """
    origin = torch.tensor(sensor_origin, dtype=points.dtype, device=points.device)
    target = torch.tensor(sensor_target, dtype=points.dtype, device=points.device)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=points.dtype, device=points.device)

    # look_at 기저 벡터 (카메라 z축 = -forward)
    forward = target - origin
    forward = forward / forward.norm()
    right = torch.cross(forward, up)
    right = right / right.norm()
    true_up = torch.cross(right, forward)

    # 카메라→세계 회전 행렬 (카메라의 x,y,z 축이 세계좌표에서 어디를 가리키는지)
    # Mitsuba convention: 카메라 -z 방향이 forward
    R = torch.stack([right, true_up, -forward], dim=1)  # (3, 3)

    # 변환: p_world = R @ p_cam + origin
    world_points = (R @ points.T).T + origin.unsqueeze(0)
    return world_points

def create_interpolator(_frames, _pointclouds, environment_pir,
                        frame_rate=30, remove_zeros=True,
                        sensor_origin=None, sensor_target=None):
    """
    Body pointclouds (from Mitsuba si.p) are already in world coordinates.
    Environment pointclouds (from PIR depth) are in camera space and need
    camera_to_world_points() transformation.
    """
    num_frames = len(_frames)
    total_time = num_frames / frame_rate
    frames = _frames.copy()
    pointclouds = _pointclouds.copy()

    # Body pointclouds from Mitsuba si.p are already in world coordinates.
    # Do NOT apply camera_to_world_points here.

    if environment_pir is not None:
        environment_pir = environment_pir.resize((64, 64), resample=Image.Resampling.BILINEAR)
        environment_pir = torch.tensor(np.array(environment_pir), dtype=torch.float32) / 255.0
        environment_points = calculate_environment_points(environment_pir)
        # Transform environment points from camera space to world coordinates
        if sensor_origin is not None and sensor_target is not None:
            environment_points = camera_to_world_points(
                environment_points, sensor_origin, sensor_target
            )
        environment_intensity = environment_pir[:, :, 1].flatten()

    def interpolator(time):
        if time < 0 or time > total_time:
            raise ValueError("Invalid time value")

        frame_index = int(time * frame_rate)
        if frame_index == num_frames:
            return frames[-1]

        t = (time * frame_rate) % 1
        frame1 = frames[frame_index]
        next_idx = min(frame_index + 1, len(frames) - 1)
        frame2 = frames[next_idx]

        pointcloud1 = pointclouds[frame_index]
        pointcloud2 = pointclouds[next_idx]

        zero_depth_frame1 = frame1[:, :, 1] == 0
        zero_depth_frame2 = frame2[:, :, 1] == 0
        zero_depth_frame1_flat = zero_depth_frame1.reshape(-1)
        zero_depth_frame2_flat = zero_depth_frame2.reshape(-1)

        frame1[zero_depth_frame1] = frame2[zero_depth_frame1]
        frame2[zero_depth_frame2] = frame1[zero_depth_frame2]
        pointcloud1[zero_depth_frame1_flat] = pointcloud2[zero_depth_frame1_flat]
        pointcloud2[zero_depth_frame2_flat] = pointcloud1[zero_depth_frame2_flat]

        interpolated_frame = frame1 * (1 - t) + frame2 * t
        interpolated_pointcloud = pointcloud1 * (1 - t) + pointcloud2 * t

        flatten_pir = interpolated_frame.reshape(-1, 3)
        intensity = flatten_pir[:, 0]
        depth = flatten_pir[:, 1]
        mask = (depth > 0.1) & (intensity > 0.1)

        if environment_pir is not None:
            combined_intensity = torch.cat((environment_intensity, intensity[mask]), dim=0)
            combined_pointcloud = torch.cat((environment_points, interpolated_pointcloud[mask]), dim=0)
        else:
            combined_intensity = intensity[mask]
            combined_pointcloud = interpolated_pointcloud[mask]

        return combined_intensity, combined_pointcloud

    return interpolator




def generate_signal_frames(body_pirs, body_auxs, envir_pir, radar_config,
                           sensor_origin=None, sensor_target=None):
    """
    Generate radar signal frames.

    sensor_origin/sensor_target: world-coordinate position of the radar sensor.
    Used for (1) environment pointcloud camera-to-world transform and
    (2) offsetting antenna TX/RX positions to the sensor location.
    """
    interpolator = create_interpolator(
        body_pirs, body_auxs, envir_pir,
        frame_rate=30,
        sensor_origin=sensor_origin,
        sensor_target=sensor_target,
    )
    total_motion_frames = len(body_pirs)
    radar = Radar(radar_config)

    total_radar_frame = int(total_motion_frames / 30 * radar.frame_per_second)
    frames = []
    for i in tqdm(range(total_radar_frame), desc="Generating radar frames"):
        frame_mimo = radar.frameMIMO(
            interpolator, i * 1.0 / radar.frame_per_second,
            sensor_origin=sensor_origin,
        )
        frames.append(frame_mimo.cpu().numpy())
    frames = np.array(frames)
    return frames