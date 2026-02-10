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
 
 
def convert_hymotion_to_smpl24(rot6d, transl):
    """Convert HY-Motion output (SMPL-H 22-joint rot6d) to standard SMPL 24-joint axis-angle.
 
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
 
 
def generate(prompt, out_dir, hymotion_output=None):
    """Generate motion from text prompt using HY-Motion and save as SMPL 24-joint format.
 
    Args:
        prompt: text description of the motion
        out_dir: output directory to save obj_diff.npz
        hymotion_output: pre-generated HY-Motion output dict with 'rot6d' and 'transl' keys.
                         If None, will attempt to use MDM fallback.
    """
    if hymotion_output is not None:
        # Use HY-Motion output directly
        print(colored('[RFGen.ObjDiff]: Converting HY-Motion output to SMPL-24 format.', 'green'))
 
        rot6d = hymotion_output['rot6d']
        transl = hymotion_output['transl']
 
        # Handle batch dimension: take first sample if batched
        if rot6d.ndim == 4:
            rot6d = rot6d[0]    # (L, J, 6)
            transl = transl[0]  # (L, 3)
 
        pose, shape, root_translation = convert_hymotion_to_smpl24(rot6d, transl)
 
        np.savez(
            os.path.join(out_dir, 'obj_diff.npz'),
            pose=pose,
            shape=shape,
            root_translation=root_translation,
            gender="male",
        )
        print(colored(f'[RFGen.ObjDiff]: Saved obj_diff.npz with pose shape {pose.shape}', 'green'))
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
    thetas_vec3 = geometry.matrix_to_euler_angles(thetas_matrix,"XYZ")
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


    

