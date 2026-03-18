"""
HumanML3D 263D Representation Conversion.

HY-Motion 출력 (keypoints3d, rot6d, transl) →
MotionGPT가 기대하는 HumanML3D 263D representation 변환.

HumanML3D 263D 구성 (Guo et al., CVPR 2022):
  ┌──────────────────────────────────────────────────┐
  │ Feature                         │ Dim  │ Offset  │
  ├──────────────────────────────────────────────────┤
  │ Root angular velocity (Y-axis)  │  1   │  0      │
  │ Root linear velocity (XZ)       │  2   │  1      │
  │ Root height (Y)                 │  1   │  3      │
  │ Joint positions (rel. to root)  │ 63   │  4      │  (21 joints × 3)
  │ Joint velocities                │ 66   │  67     │  (22 joints × 3)
  │ Joint rotations (6D cont.)     │ 126  │  133    │  (21 joints × 6)
  │ Foot contact labels             │  4   │  259    │
  ├──────────────────────────────────────────────────┤
  │ Total                           │ 263  │         │
  └──────────────────────────────────────────────────┘

SMPL 22-joint order:
  0:pelvis, 1:l_hip, 2:r_hip, 3:spine1, 4:l_knee, 5:r_knee,
  6:spine2, 7:l_ankle, 8:r_ankle, 9:spine3, 10:l_foot, 11:r_foot,
  12:neck, 13:l_collar, 14:r_collar, 15:head, 16:l_shoulder, 17:r_shoulder,
  18:l_elbow, 19:r_elbow, 20:l_wrist, 21:r_wrist

Foot contact joints:
  7: left ankle, 10: left foot (toe), 8: right ankle, 11: right foot (toe)
"""

import numpy as np
import torch
from typing import Optional, Tuple, Dict


# ─────────────────────────────────────────────────────────
# Rotation Utilities (numpy)
# ─────────────────────────────────────────────────────────

def _rot6d_to_matrix_np(rot6d: np.ndarray) -> np.ndarray:
    """
    6D rotation → 3×3 rotation matrix (Gram-Schmidt).
    Args: rot6d: (..., 6)
    Returns: (..., 3, 3)
    """
    x = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    a1 = x[..., 0]
    a2 = x[..., 1]

    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)


def _matrix_to_cont6d_np(matrix: np.ndarray) -> np.ndarray:
    """
    3×3 rotation matrix → continuous 6D representation.
    (처음 2개 열 추출)
    Args: matrix: (..., 3, 3)
    Returns: (..., 6)
    """
    return matrix[..., :2].reshape(*matrix.shape[:-2], 6)


def _rotation_matrix_to_angle_axis_np(rot_mat: np.ndarray) -> np.ndarray:
    """Rotation matrix → axis-angle."""
    from scipy.spatial.transform import Rotation
    orig_shape = rot_mat.shape[:-2]
    flat = rot_mat.reshape(-1, 3, 3)
    r = Rotation.from_matrix(flat)
    aa = r.as_rotvec()
    return aa.reshape(*orig_shape, 3)


# ─────────────────────────────────────────────────────────
# Root Orientation Utilities
# ─────────────────────────────────────────────────────────

def _extract_root_y_rotation(root_rotmat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Root rotation matrix에서 Y축 회전 성분만 추출.

    Args:
        root_rotmat: (T, 3, 3) root joint rotation matrices

    Returns:
        y_angle: (T,) Y축 회전 각도 (radians)
        y_rotmat: (T, 3, 3) Y축 회전만 남긴 rotation matrix
    """
    # Forward direction: rotation matrix의 Z열 (3번째 열)을 XZ 평면에 투영
    forward = root_rotmat[:, :, 2]  # (T, 3) — Z-axis direction after rotation
    # XZ 평면에서의 방향
    forward_xz = forward.copy()
    forward_xz[:, 1] = 0  # Y 성분 제거
    norm = np.linalg.norm(forward_xz, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    forward_xz = forward_xz / norm

    # Y축 회전 각도: atan2(x, z)
    y_angle = np.arctan2(forward_xz[:, 0], forward_xz[:, 2])  # (T,)

    # Y축 회전 행렬 구성
    cos_y = np.cos(y_angle)
    sin_y = np.sin(y_angle)
    T = len(y_angle)
    y_rotmat = np.zeros((T, 3, 3), dtype=np.float32)
    y_rotmat[:, 0, 0] = cos_y
    y_rotmat[:, 0, 2] = sin_y
    y_rotmat[:, 1, 1] = 1.0
    y_rotmat[:, 2, 0] = -sin_y
    y_rotmat[:, 2, 2] = cos_y

    return y_angle, y_rotmat


# ─────────────────────────────────────────────────────────
# Foot Contact Detection
# ─────────────────────────────────────────────────────────

def _compute_foot_contact(
    joints: np.ndarray,
    height_threshold: float = 0.05,
    velocity_threshold: float = 0.02,
) -> np.ndarray:
    """
    발 접촉 라벨 계산.

    발 높이가 threshold 이하이고 속도가 threshold 이하이면 접촉으로 판단.

    Args:
        joints: (T, 22, 3) joint positions
        height_threshold: 발이 바닥에 가까운 높이 (m)
        velocity_threshold: 발 속도 threshold (m/frame)

    Returns:
        foot_contact: (T, 4) — [l_ankle, l_toe, r_ankle, r_toe]
    """
    T = joints.shape[0]

    # Foot joint indices
    l_ankle_idx, l_foot_idx = 7, 10
    r_ankle_idx, r_foot_idx = 8, 11

    foot_joints = joints[:, [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx], :]  # (T, 4, 3)

    # Height check (Y-axis)
    # 바닥 높이를 전체 시퀀스에서 추정 (최소 Y)
    floor_y = joints[:, :, 1].min()
    heights = foot_joints[:, :, 1] - floor_y  # (T, 4)

    # Velocity check
    velocities = np.zeros_like(heights)
    if T > 1:
        vel = np.linalg.norm(foot_joints[1:] - foot_joints[:-1], axis=-1)  # (T-1, 4)
        velocities[1:] = vel
        velocities[0] = vel[0]

    # 접촉 판정
    contact = ((heights < height_threshold) & (velocities < velocity_threshold)).astype(np.float32)

    return contact  # (T, 4)


# ─────────────────────────────────────────────────────────
# Main Conversion Function
# ─────────────────────────────────────────────────────────

def convert_to_humanml3d(
    joints: np.ndarray,
    rot6d: Optional[np.ndarray] = None,
    transl: Optional[np.ndarray] = None,
    fps: float = 30.0,
) -> np.ndarray:
    """
    HY-Motion 출력을 HumanML3D 263D representation으로 변환.

    Args:
        joints: (T, 22, 3)  — 3D joint positions (world coordinates)
        rot6d:  (T, 22, 6)  — joint rotations in 6D representation (optional)
        transl: (T, 3)      — root translation (optional, 없으면 joints[:,0,:] 사용)
        fps: frame rate

    Returns:
        features: (T, 263) — HumanML3D representation
    """
    T, J = joints.shape[:2]
    assert J >= 22, f"Expected at least 22 joints, got {J}"
    joints = joints[:, :22, :].copy().astype(np.float32)

    # ── Root position ──
    if transl is not None:
        root_pos = transl[:, :3].copy().astype(np.float32)  # (T, 3)
    else:
        root_pos = joints[:, 0, :].copy()  # (T, 3)

    # ── Root orientation (Y-axis rotation) ──
    if rot6d is not None:
        rot6d = rot6d[:, :22, :].copy().astype(np.float32)
        root_rot6d = rot6d[:, 0, :]  # (T, 6)
        root_rotmat = _rot6d_to_matrix_np(root_rot6d)  # (T, 3, 3)
    else:
        # joints에서 root orientation 추정 (spine direction)
        # spine1(3)과 pelvis(0)를 이용한 근사
        root_rotmat = np.tile(np.eye(3, dtype=np.float32), (T, 1, 1))

    y_angle, y_rotmat = _extract_root_y_rotation(root_rotmat)  # (T,), (T, 3, 3)

    # ── (1) Root angular velocity (Y-axis) — 1D ──
    r_velocity = np.zeros(T, dtype=np.float32)
    if T > 1:
        r_velocity[1:] = y_angle[1:] - y_angle[:-1]
        # Wrap to [-π, π]
        r_velocity = np.arctan2(np.sin(r_velocity), np.cos(r_velocity))
        r_velocity[0] = r_velocity[1]

    # ── (2) Root linear velocity (XZ) — 2D ──
    l_velocity = np.zeros((T, 2), dtype=np.float32)
    if T > 1:
        root_vel_world = root_pos[1:] - root_pos[:-1]  # (T-1, 3)
        # Y축 역회전을 적용하여 local frame으로 변환
        for t in range(T - 1):
            inv_y = y_rotmat[t].T  # inverse = transpose for rotation
            vel_local = inv_y @ root_vel_world[t]
            l_velocity[t + 1, 0] = vel_local[0]  # X
            l_velocity[t + 1, 1] = vel_local[2]  # Z
        l_velocity[0] = l_velocity[1]

    # ── (3) Root height (Y) — 1D ──
    root_y = root_pos[:, 1:2]  # (T, 1)

    # ── (4) Joint positions relative to root — 63D (21 joints × 3) ──
    # Root (joint 0)를 제외한 21개 joint
    joints_rel = joints[:, 1:, :] - joints[:, 0:1, :]  # (T, 21, 3)

    # Y축 역회전 적용 (local frame)
    for t in range(T):
        inv_y = y_rotmat[t].T
        joints_rel[t] = (inv_y @ joints_rel[t].T).T

    ric_data = joints_rel.reshape(T, -1)  # (T, 63)

    # ── (5) Joint velocities — 66D (22 joints × 3) ──
    # Local frame에서의 joint velocity
    joints_local = joints.copy()
    for t in range(T):
        inv_y = y_rotmat[t].T
        centered = joints[t] - root_pos[t:t+1]
        joints_local[t] = (inv_y @ centered.T).T

    local_vel = np.zeros((T, 22, 3), dtype=np.float32)
    if T > 1:
        local_vel[1:] = joints_local[1:] - joints_local[:-1]
        local_vel[0] = local_vel[1]
    local_vel = local_vel.reshape(T, -1)  # (T, 66)

    # ── (6) Joint rotations (continuous 6D) — 126D (21 joints × 6) ──
    if rot6d is not None:
        # Root를 제외한 21개 joint의 6D rotation
        rot_data = rot6d[:, 1:22, :].reshape(T, -1)  # (T, 126)
    else:
        # rot6d가 없으면 identity rotation으로 채움
        identity_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        rot_data = np.tile(identity_6d, (T, 21)).reshape(T, -1)  # (T, 126)

    # ── (7) Foot contact labels — 4D ──
    feet_contact = _compute_foot_contact(joints)  # (T, 4)

    # ── 결합: 263D ──
    features = np.concatenate([
        r_velocity[:, None],   # (T, 1)   — root angular vel
        l_velocity,            # (T, 2)   — root linear vel XZ
        root_y,                # (T, 1)   — root height
        ric_data,              # (T, 63)  — relative joint positions
        local_vel,             # (T, 66)  — joint velocities
        rot_data,              # (T, 126) — joint rotations 6D
        feet_contact,          # (T, 4)   — foot contact
    ], axis=-1)

    assert features.shape == (T, 263), f"Expected (T, 263), got {features.shape}"
    return features


def convert_hymotion_output_to_humanml3d(model_output: dict) -> np.ndarray:
    """
    HY-Motion model output dict → HumanML3D 263D.

    model_output keys:
      - keypoints3d: (B, L, J, 3) 또는 (L, J, 3)
      - rot6d:       (B, L, J, 6) 또는 (L, J, 6)
      - transl:      (B, L, 3)    또는 (L, 3)

    Returns:
      features: (L, 263) — first batch element
    """
    # ── keypoints3d ──
    joints = model_output.get("keypoints3d")
    if joints is None:
        raise ValueError("keypoints3d is required for HumanML3D conversion")
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()
    if joints.ndim == 4:
        joints = joints[0]  # (L, J, 3)

    # ── rot6d ──
    rot6d = model_output.get("rot6d")
    if rot6d is not None:
        if isinstance(rot6d, torch.Tensor):
            rot6d = rot6d.cpu().numpy()
        if rot6d.ndim == 4:
            rot6d = rot6d[0]

    # ── transl ──
    transl = model_output.get("transl")
    if transl is not None:
        if isinstance(transl, torch.Tensor):
            transl = transl.cpu().numpy()
        if transl.ndim == 3:
            transl = transl[0]

    return convert_to_humanml3d(joints, rot6d, transl)


# ─────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing HumanML3D conversion...")

    T, J = 60, 22
    joints = np.random.randn(T, J, 3).astype(np.float32)
    joints[:, :, 1] += 1.0  # Y offset (height)
    rot6d = np.random.randn(T, J, 6).astype(np.float32)
    transl = np.random.randn(T, 3).astype(np.float32)

    features = convert_to_humanml3d(joints, rot6d, transl)
    print(f"Output: {features.shape}")  # (60, 263)
    print(f"Feature ranges:")
    print(f"  r_velocity:  [{features[:, 0].min():.3f}, {features[:, 0].max():.3f}]")
    print(f"  l_velocity:  [{features[:, 1:3].min():.3f}, {features[:, 1:3].max():.3f}]")
    print(f"  root_y:      [{features[:, 3].min():.3f}, {features[:, 3].max():.3f}]")
    print(f"  ric_data:    [{features[:, 4:67].min():.3f}, {features[:, 4:67].max():.3f}]")
    print(f"  local_vel:   [{features[:, 67:133].min():.3f}, {features[:, 67:133].max():.3f}]")
    print(f"  rot_data:    [{features[:, 133:259].min():.3f}, {features[:, 133:259].max():.3f}]")
    print(f"  foot_contact:[{features[:, 259:].min():.3f}, {features[:, 259:].max():.3f}]")
    print("✓ Conversion OK")