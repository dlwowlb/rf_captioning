import sys
import os
import subprocess
from termcolor import colored
import numpy as np
import torch
import torch.nn.functional as F


def _rot6d_to_rotation_matrix(rot6d):
    """Convert 6D rotation representation to rotation matrix (3x3).

    Matches HY-Motion's convention where the 6D vector is interpreted as a 3x2 matrix
    (two column vectors), and the output rotation matrix has b1, b2, b3 as columns.

    Based on Zhou et al., CVPR 2019.
    """
    x = rot6d.view(*rot6d.shape[:-1], 3, 2)
    a1 = x[..., 0]
    a2 = x[..., 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - torch.einsum("...i,...i->...", b1, a2).unsqueeze(-1) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _matrix_to_axis_angle(matrix):
    """Convert rotation matrix to axis-angle representation.

    Uses quaternion as intermediate to avoid gimbal lock issues.
    """
    quat = _matrix_to_quaternion(matrix)
    return _quaternion_to_axis_angle(quat)


def _matrix_to_quaternion(matrix):
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack([
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ], dim=-1),
            min=0.0,
        )
    )
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    # Standardize: ensure w >= 0
    return torch.where(out[..., 0:1] < 0, -out, out)


def _quaternion_to_axis_angle(quaternions):
    """Convert quaternion (w, x, y, z) to axis-angle."""
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def _batch_rodrigues(rot_vecs, epsilon=1e-8):
    """Batch rodrigues: axis-angle (N, 3) -> rotation matrix (N, 3, 3)."""
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view(batch_size, 3, 3)

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def _forward_kinematics(rot_mats, joints, parents):
    """Compute forward kinematics to get posed joint positions.

    Args:
        rot_mats: (B, J, 3, 3) local rotation matrices
        joints: (J, 3) rest-pose joint positions
        parents: (J,) parent index for each joint

    Returns:
        posed_joints: (B, J, 3) world-space joint positions
    """
    batch_size = rot_mats.shape[0]
    num_joints = rot_mats.shape[1]
    device = rot_mats.device
    dtype = rot_mats.dtype

    # Relative joint positions
    rel_joints = joints.clone()
    rel_joints[1:] = joints[1:] - joints[parents[1:]]

    # Build 4x4 transform matrices
    transforms = torch.zeros(batch_size, num_joints, 4, 4, device=device, dtype=dtype)
    transforms[..., :3, :3] = rot_mats
    transforms[..., :3, 3] = rel_joints.unsqueeze(0).expand(batch_size, -1, -1)
    transforms[..., 3, 3] = 1.0

    # Chain transforms from root to leaves
    chain = [transforms[:, 0]]
    for i in range(1, num_joints):
        parent_idx = parents[i].item() if hasattr(parents[i], 'item') else int(parents[i])
        curr = torch.bmm(chain[parent_idx], transforms[:, i])
        chain.append(curr)

    global_transforms = torch.stack(chain, dim=1)  # (B, J, 4, 4)
    posed_joints = global_transforms[..., :3, 3]   # (B, J, 3)
    return posed_joints


# SMPL-H joint hierarchy: parent index for each of the 52 joints
# Joints 0-21: body, Joints 22-36: left hand, Joints 37-51: right hand
SMPLH_PARENTS = [
    -1,  # 0: Pelvis (root)
    0,   # 1: L_Hip -> Pelvis
    0,   # 2: R_Hip -> Pelvis
    0,   # 3: Spine1 -> Pelvis
    1,   # 4: L_Knee -> L_Hip
    2,   # 5: R_Knee -> R_Hip
    3,   # 6: Spine2 -> Spine1
    4,   # 7: L_Ankle -> L_Knee
    5,   # 8: R_Ankle -> R_Knee
    6,   # 9: Spine3 -> Spine2
    7,   # 10: L_Foot -> L_Ankle
    8,   # 11: R_Foot -> R_Ankle
    9,   # 12: Neck -> Spine3
    9,   # 13: L_Collar -> Spine3
    9,   # 14: R_Collar -> Spine3
    12,  # 15: Head -> Neck
    13,  # 16: L_Shoulder -> L_Collar
    14,  # 17: R_Shoulder -> R_Collar
    16,  # 18: L_Elbow -> L_Shoulder
    17,  # 19: R_Elbow -> R_Shoulder
    18,  # 20: L_Wrist -> L_Elbow
    19,  # 21: R_Wrist -> R_Elbow
    # Left hand (parent = L_Wrist = 20)
    20,  # 22: L_Index1
    22,  # 23: L_Index2
    23,  # 24: L_Index3
    20,  # 25: L_Middle1
    25,  # 26: L_Middle2
    26,  # 27: L_Middle3
    20,  # 28: L_Pinky1
    28,  # 29: L_Pinky2
    29,  # 30: L_Pinky3
    20,  # 31: L_Ring1
    31,  # 32: L_Ring2
    32,  # 33: L_Ring3
    20,  # 34: L_Thumb1
    34,  # 35: L_Thumb2
    35,  # 36: L_Thumb3
    # Right hand (parent = R_Wrist = 21)
    21,  # 37: R_Index1
    37,  # 38: R_Index2
    38,  # 39: R_Index3
    21,  # 40: R_Middle1
    40,  # 41: R_Middle2
    41,  # 42: R_Middle3
    21,  # 43: R_Pinky1
    43,  # 44: R_Pinky2
    44,  # 45: R_Pinky3
    21,  # 46: R_Ring1
    46,  # 47: R_Ring2
    47,  # 48: R_Ring3
    21,  # 49: R_Thumb1
    49,  # 50: R_Thumb2
    50,  # 51: R_Thumb3
]

# SMPL-H rest-pose joint offsets (relative to parent) for hand joints
# These are approximate offsets in meters for a neutral body shape.
# Finger joints branch from the wrist with small offsets.
HAND_JOINT_OFFSETS = {
    # Left hand (relative to L_Wrist)
    22: [0.0975, -0.0076, -0.0194],   # L_Index1
    23: [0.0267, -0.0014, -0.0008],   # L_Index2
    24: [0.0198, -0.0004, 0.0003],    # L_Index3
    25: [0.0939, -0.0062, 0.0101],    # L_Middle1
    26: [0.0300, -0.0014, 0.0003],    # L_Middle2
    27: [0.0218, -0.0005, 0.0004],    # L_Middle3
    28: [0.0712, -0.0040, 0.0478],    # L_Pinky1
    29: [0.0199, -0.0010, 0.0005],    # L_Pinky2
    30: [0.0156, -0.0002, 0.0003],    # L_Pinky3
    31: [0.0864, -0.0053, 0.0286],    # L_Ring1
    32: [0.0264, -0.0014, 0.0004],    # L_Ring2
    33: [0.0202, -0.0004, 0.0003],    # L_Ring3
    34: [0.0393, 0.0084, -0.0283],    # L_Thumb1
    35: [0.0338, -0.0060, -0.0101],   # L_Thumb2
    36: [0.0257, -0.0013, -0.0029],   # L_Thumb3
    # Right hand (relative to R_Wrist) - mirrored X
    37: [0.0975, 0.0076, 0.0194],     # R_Index1
    38: [0.0267, 0.0014, 0.0008],     # R_Index2
    39: [0.0198, 0.0004, -0.0003],    # R_Index3
    40: [0.0939, 0.0062, -0.0101],    # R_Middle1
    41: [0.0300, 0.0014, -0.0003],    # R_Middle2
    42: [0.0218, 0.0005, -0.0004],    # R_Middle3
    43: [0.0712, 0.0040, -0.0478],    # R_Pinky1
    44: [0.0199, 0.0010, -0.0005],    # R_Pinky2
    45: [0.0156, 0.0002, -0.0003],    # R_Pinky3
    46: [0.0864, 0.0053, -0.0286],    # R_Ring1
    47: [0.0264, 0.0014, -0.0004],    # R_Ring2
    48: [0.0202, 0.0004, -0.0003],    # R_Ring3
    49: [0.0393, -0.0084, 0.0283],    # R_Thumb1
    50: [0.0338, 0.0060, 0.0101],     # R_Thumb2
    51: [0.0257, 0.0013, 0.0029],     # R_Thumb3
}


def convert_hymotion_to_smpl24(rot6d, transl):
    """Convert HY-Motion output to standard SMPL 24-joint axis-angle.

    HY-Motion outputs 22 body joints in rot6d format (same ordering as SMPL body joints 0-21).
    Standard SMPL expects 24 joints: joints 0-21 (body) + joint 22 (L_Hand) + joint 23 (R_Hand).

    We map joints 0-21 directly and set joints 22-23 (hand roots) to identity rotation.

    Args:
        rot6d: (num_frames, num_joints, 6) - HY-Motion rot6d output, num_joints is 22 or 52
        transl: (num_frames, 3) - root translation

    Returns:
        pose: (num_frames, 72) - standard SMPL axis-angle pose (24 joints x 3)
        shape: (10,) - zero body shape
        root_translation: (num_frames, 3) - root translation
    """
    if isinstance(rot6d, np.ndarray):
        rot6d = torch.from_numpy(rot6d).float()
    if isinstance(transl, np.ndarray):
        transl = torch.from_numpy(transl).float()

    num_frames = rot6d.shape[0]

    # Take only the first 22 body joints (discard hand finger joints if present)
    body_rot6d = rot6d[:, :22, :]  # (num_frames, 22, 6)

    # Convert 6D rotation to rotation matrix, then to axis-angle
    rot_matrices = _rot6d_to_rotation_matrix(body_rot6d)  # (num_frames, 22, 3, 3)
    body_aa = _matrix_to_axis_angle(rot_matrices)  # (num_frames, 22, 3)

    # Create identity rotation (zero axis-angle) for hand root joints (L_Hand=22, R_Hand=23)
    hand_aa = torch.zeros(num_frames, 2, 3, dtype=body_aa.dtype, device=body_aa.device)

    # Concatenate: 22 body joints + 2 hand joints = 24 SMPL joints
    smpl_aa = torch.cat([body_aa, hand_aa], dim=1)  # (num_frames, 24, 3)

    # Flatten to (num_frames, 72)
    pose = smpl_aa.reshape(num_frames, -1).cpu().numpy()
    shape = np.zeros(10)
    root_translation = transl.cpu().numpy()

    return pose, shape, root_translation


def convert_hymotion_to_smplh52(rot6d, transl):
    """Convert full HY-Motion SMPL-H 52-joint rot6d output to axis-angle.

    Preserves all 52 joints including 30 hand joints.

    Args:
        rot6d: (num_frames, 52, 6) - full SMPL-H rot6d
        transl: (num_frames, 3) - root translation

    Returns:
        pose_smplh: (num_frames, 156) - SMPL-H axis-angle (52 joints x 3)
    """
    if isinstance(rot6d, np.ndarray):
        rot6d = torch.from_numpy(rot6d).float()

    num_joints = rot6d.shape[1]
    rot_matrices = _rot6d_to_rotation_matrix(rot6d)      # (N, J, 3, 3)
    aa = _matrix_to_axis_angle(rot_matrices)              # (N, J, 3)
    return aa.reshape(rot6d.shape[0], -1).cpu().numpy()   # (N, J*3)


def compute_hand_keypoints3d(pose_smplh, transl, body_keypoints3d=None):
    """Compute 52-joint keypoints3d via forward kinematics from SMPL-H axis-angle pose.

    Uses body_keypoints3d for the 22 body joints if provided (from SMPL_Layer),
    and computes hand joint positions via FK from the wrist.

    Args:
        pose_smplh: (num_frames, 156) - SMPL-H axis-angle (52 joints x 3)
        transl: (num_frames, 3) - root translation
        body_keypoints3d: (num_frames, 24, 3) or None - body joint positions from SMPL_Layer

    Returns:
        keypoints3d: (num_frames, 52, 3) - all joint 3D positions
    """
    num_frames = pose_smplh.shape[0]
    device = 'cpu'

    if isinstance(pose_smplh, np.ndarray):
        pose_smplh = torch.from_numpy(pose_smplh).float()
    if isinstance(transl, np.ndarray):
        transl = torch.from_numpy(transl).float()

    # Reshape to (N, 52, 3) axis-angle per joint
    num_joints = pose_smplh.shape[1] // 3
    aa_per_joint = pose_smplh.reshape(num_frames, num_joints, 3)

    # Convert all joints to rotation matrices
    rot_mats = _batch_rodrigues(aa_per_joint.reshape(-1, 3)).reshape(num_frames, num_joints, 3, 3)

    # Build rest-pose joint positions for all 52 joints
    # We need approximate rest-pose positions. For body joints (0-21), use zeros
    # (positions come from SMPL model). For hand joints (22-51), use HAND_JOINT_OFFSETS.
    parents = torch.tensor(SMPLH_PARENTS[:num_joints], dtype=torch.long)

    # Build approximate rest-pose from offsets
    j_rest = torch.zeros(num_joints, 3, dtype=torch.float32)

    # Set hand joint offsets
    for jidx, offset in HAND_JOINT_OFFSETS.items():
        if jidx < num_joints:
            j_rest[jidx] = torch.tensor(offset)

    # Compute full FK
    posed_joints = _forward_kinematics(rot_mats, j_rest, parents)  # (N, 52, 3)

    # If we have body keypoints from SMPL_Layer, overlay them for better accuracy
    if body_keypoints3d is not None:
        if isinstance(body_keypoints3d, np.ndarray):
            body_keypoints3d = torch.from_numpy(body_keypoints3d).float()
        body_num = body_keypoints3d.shape[1]  # 24

        # Use SMPL body joint positions for joints 0-21 (or 0-23)
        # and offset hand joints relative to wrists
        posed_joints_out = torch.zeros(num_frames, num_joints, 3)
        posed_joints_out[:, :min(body_num, 22)] = body_keypoints3d[:, :min(body_num, 22)]

        # Compute hand joint positions relative to wrist using FK offsets
        # Left hand: joints 22-36, parent chain from L_Wrist (20)
        # Right hand: joints 37-51, parent chain from R_Wrist (21)
        l_wrist_pos = body_keypoints3d[:, 20:21, :]  # (N, 1, 3)
        r_wrist_pos = body_keypoints3d[:, 21:22, :]  # (N, 1, 3)

        # FK delta from rest-pose wrist to each hand joint
        if num_joints > 22:
            fk_wrist_l = posed_joints[:, 20:21, :]
            fk_wrist_r = posed_joints[:, 21:22, :]

            for jidx in range(22, min(37, num_joints)):
                delta = posed_joints[:, jidx:jidx+1, :] - fk_wrist_l
                posed_joints_out[:, jidx:jidx+1, :] = l_wrist_pos + delta

            for jidx in range(37, min(52, num_joints)):
                delta = posed_joints[:, jidx:jidx+1, :] - fk_wrist_r
                posed_joints_out[:, jidx:jidx+1, :] = r_wrist_pos + delta

        # Add translation
        posed_joints_out = posed_joints_out + transl.unsqueeze(1)
        return posed_joints_out.cpu().numpy()
    else:
        # Pure FK result + translation
        posed_joints = posed_joints + transl.unsqueeze(1)
        return posed_joints.cpu().numpy()


def generate(prompt, out_dir, hymotion_output=None):
    """Generate motion from text prompt and save as SMPL format for RF-Genesis.

    When hymotion_output is provided, saves both 24-joint SMPL pose (for ray tracing)
    and full 52-joint SMPL-H data (hand poses + keypoints) for visualization/simulation.

    Args:
        prompt: text description of the motion
        out_dir: output directory to save obj_diff.npz
        hymotion_output: pre-generated HY-Motion output dict with keys:
            - 'rot6d': (B, L, J, 6) or (L, J, 6) - joint rotations
            - 'transl': (B, L, 3) or (L, 3) - root translation
            - 'keypoints3d': (B, L, J, 3) optional - joint 3D positions
    """
    if hymotion_output is not None:
        print(colored('[RFGen.ObjDiff]: Converting HY-Motion output to SMPL format.', 'green'))

        rot6d = hymotion_output['rot6d']
        transl = hymotion_output['transl']
        keypoints3d = hymotion_output.get('keypoints3d', None)

        # Handle batch dimension: take first sample if batched
        if rot6d.ndim == 4:
            rot6d = rot6d[0]        # (L, J, 6)
            transl = transl[0]      # (L, 3)
            if keypoints3d is not None:
                keypoints3d = keypoints3d[0]  # (L, J, 3)

        if isinstance(rot6d, torch.Tensor):
            rot6d_np = rot6d.cpu()
        else:
            rot6d_np = rot6d

        # 24-joint SMPL pose for SMPL_Layer ray tracing
        pose, shape, root_translation = convert_hymotion_to_smpl24(rot6d, transl)

        # Full 52-joint SMPL-H pose (preserving hand articulation)
        num_joints = rot6d.shape[1]
        save_dict = dict(
            pose=pose,                      # (N, 72) - 24-joint SMPL for ray tracing
            shape=shape,                    # (10,)
            root_translation=root_translation,  # (N, 3)
            gender="male",
        )

        if num_joints >= 52:
            pose_smplh = convert_hymotion_to_smplh52(rot6d, transl)
            save_dict['pose_smplh'] = pose_smplh  # (N, 156) - full 52-joint SMPL-H

        # Save keypoints3d if available from HY-Motion
        if keypoints3d is not None:
            if isinstance(keypoints3d, torch.Tensor):
                kp = keypoints3d.cpu().numpy()
            else:
                kp = keypoints3d
            save_dict['keypoints3d'] = kp  # (N, 52, 3) - all joint positions

        np.savez(os.path.join(out_dir, 'obj_diff.npz'), **save_dict)
        print(colored(
            f'[RFGen.ObjDiff]: Saved obj_diff.npz - pose:{pose.shape}'
            + (f', pose_smplh:{pose_smplh.shape}' if num_joints >= 52 else '')
            + (f', keypoints3d:{kp.shape}' if keypoints3d is not None else ''),
            'green',
        ))
    else:
        # Fallback: use original MDM pipeline
        print(colored('[RFGen.ObjDiff]: No HY-Motion output provided, using MDM fallback.', 'yellow'))
        _generate_mdm(prompt, out_dir)


def _generate_mdm(prompt, out_dir):
    """Original MDM-based motion generation (fallback)."""
    from scipy.spatial.transform import Rotation as R

    def euler_to_axis_angle(euler_angles):
        axis_angle_params = np.zeros_like(euler_angles)
        for i in range(euler_angles.shape[0]):
            for j in range(euler_angles.shape[1]):
                euler = euler_angles[i, j]
                r = R.from_euler('xyz', euler)
                axis_angle = r.as_rotvec()
                axis_angle_params[i, j] = axis_angle
        return axis_angle_params

    os.chdir("ext/mdm/")
    subprocess.run(
        ['python', '-m', 'sample.generate_rfgen', '--model_path', './save/humanml_trans_enc_512/model000200000.pt',
         '--text_prompt', prompt,
         '--output_dir', "../../"+out_dir,
         '--num_samples', '1', '--num_repetitions', '1'])
    os.chdir("../..")

    # Process MDM output
    filename = out_dir + "/obj_diff_raw.npy"
    print(colored("---[RFGen.ObjDiff]:Runing SMPLify, it may take a few minutes.---", 'yellow'))
    print(colored("---[RFGen.ObjDiff]:This may be optimized in future updates.---", 'yellow'))

    sys.path.append("ext/mdm/")
    from visualize.vis_utils import joints2smpl, npy2obj
    import utils.rotation_conversions as geometry

    data = np.load(filename, allow_pickle=True)
    motion = data[None][0]['motion'].transpose(0, 3, 1, 2)

    num_frames = motion.shape[1]
    device = '0'
    cuda = True

    os.chdir("ext/mdm")
    j2s = joints2smpl(num_frames=num_frames, device_id=device, cuda=cuda)
    os.chdir("../..")

    motion_tensor, opt_dict = j2s.joint2smpl(motion[0])
    thetas = motion_tensor[0, :-1, :, :num_frames]
    root_translation = motion_tensor[0, -1, :3, :].cpu().numpy().transpose(1, 0)

    thetas_matrix = thetas.transpose(2, 0).transpose(1, 2)
    thetas_matrix = geometry.rotation_6d_to_matrix(thetas_matrix)
    thetas_vec3 = geometry.matrix_to_euler_angles(thetas_matrix, "XYZ")
    thetas_vec3 = thetas_vec3.cpu().numpy()
    final_thetas = euler_to_axis_angle(thetas_vec3)
    smpl_params = final_thetas.reshape(final_thetas.shape[0], -1)

    shape_params = np.zeros(10)
    np.savez(
        out_dir + '/obj_diff.npz',
        pose=smpl_params,
        shape=shape_params,
        root_translation=root_translation,
        gender="male",
    )
    

