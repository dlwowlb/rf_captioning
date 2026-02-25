import numpy as np
import mitsuba as mi
import drjit as dr
import torch

torch.set_default_device('cuda')

# ============================================================================
# SMPL-X Body Model Layer (52 joints: 22 body + 15 left hand + 15 right hand)
#
# Uses the official smplx library from MPI-IS:
#   https://github.com/vchoutas/smplx
#
# SMPL-X provides full body + hand + face articulation.
# We use 52 joints (body + hands, no face) to match HY-Motion output format.
# ============================================================================

_SMPLX_BODY_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]

_SMPLX_LEFT_HAND_JOINT_NAMES = [
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
]

_SMPLX_RIGHT_HAND_JOINT_NAMES = [
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
]

SMPLX_JOINT_NAMES = (
    _SMPLX_BODY_JOINT_NAMES
    + _SMPLX_LEFT_HAND_JOINT_NAMES
    + _SMPLX_RIGHT_HAND_JOINT_NAMES
)

NUM_BODY_JOINTS = 22
NUM_HAND_JOINTS = 15
NUM_SMPLX_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS  # 52


def get_smplx_layer(model_path='../models/smplx_models', gender='neutral'):
    """
    Load SMPL-X body model using the official smplx library.

    The smplx library expects a directory structure like:
        model_path/
            smplx/
                SMPLX_NEUTRAL.npz (or .pkl)
                SMPLX_MALE.npz
                SMPLX_FEMALE.npz

    Args:
        model_path: Path to the directory containing smplx model files
        gender: 'neutral', 'male', or 'female'

    Returns:
        SMPL-X model instance on CUDA
    """
    try:
        import smplx as smplx_lib
    except ImportError:
        raise ImportError(
            "smplx library not found. Install with: pip install smplx\n"
            "Download models from: https://smpl-x.is.tue.mpg.de/"
        )

    model = smplx_lib.create(
        model_path=model_path,
        model_type='smplx',
        gender=gender,
        use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=False,       # We provide full hand rotations, not PCA
        flat_hand_mean=True,  # Start from flat hand (identity rotation)
    )
    return model.cuda()


# Backward-compatible alias
get_smpl_layer = get_smplx_layer


def apply_translation(vertices, translation_matrix):
    """Apply a 4x4 translation matrix to vertices."""
    ones = torch.ones(vertices.shape[0], 1)
    homogeneous_vertices = torch.cat([vertices, ones], dim=1)
    transformed_vertices = torch.matmul(homogeneous_vertices, translation_matrix.T)
    transformed_vertices_3d = transformed_vertices[:, :3]
    return transformed_vertices_3d


def _parse_smplx_pose(pose_params, num_pose_values):
    """
    Parse pose parameters into SMPL-X components.

    Supports multiple formats:
      - 156D: SMPL-H format (1 global_orient + 21 body + 15 lhand + 15 rhand) * 3
      - 72D:  SMPL-24 format (1 global_orient + 23 body joints) * 3
              -> pad hands with zeros (flat hand)
      - 165D: Full SMPL-X (1 global + 21 body + 3 jaw + 2 eye + 15 lhand + 15 rhand) * 3
              -> extract relevant parts

    Returns:
        global_orient: (1, 3) axis-angle
        body_pose: (1, 21*3) axis-angle
        left_hand_pose: (1, 15*3) axis-angle
        right_hand_pose: (1, 15*3) axis-angle
    """
    pose = pose_params.float()

    if num_pose_values >= 156:
        # SMPL-H / SMPL-X compatible: 22 body + 30 hand = 52 joints
        global_orient = pose[:, :3].reshape(1, 3)
        body_pose = pose[:, 3:66].reshape(1, 63)        # joints 1-21
        left_hand_pose = pose[:, 66:111].reshape(1, 45)  # joints 22-36
        right_hand_pose = pose[:, 111:156].reshape(1, 45) # joints 37-51
    elif num_pose_values >= 72:
        # SMPL-24 format: extract body, zero hands
        global_orient = pose[:, :3].reshape(1, 3)
        body_pose = pose[:, 3:66].reshape(1, 63)
        left_hand_pose = torch.zeros(1, 45, device=pose.device, dtype=pose.dtype)
        right_hand_pose = torch.zeros(1, 45, device=pose.device, dtype=pose.dtype)
    else:
        raise ValueError(
            f"Unsupported pose dimension: {num_pose_values}. "
            f"Expected 72 (SMPL-24), 156 (SMPL-H/52), or 165 (SMPL-X full)."
        )

    return global_orient, body_pose, left_hand_pose, right_hand_pose


def call_smplx_layer(pose_params, shape_params, body_model,
                     need_face=False, transform=None):
    """
    Run SMPL-X forward pass to get vertices (and optionally faces).

    Args:
        pose_params: (1, D) axis-angle pose; D = 156 (52*3) for full SMPL-H,
                     or 72 (24*3) for basic SMPL (hands will be flat)
        shape_params: (1, 10) body shape betas
        body_model: SMPL-X model from get_smplx_layer()
        need_face: if True, also return face indices
        transform: optional 4x4 translation matrix

    Returns:
        vertices_mi: Mitsuba TensorXf of vertex positions
        (optionally) faces_mi: Mitsuba TensorXf of face indices
    """
    num_pose_values = pose_params.shape[-1]
    global_orient, body_pose, left_hand_pose, right_hand_pose = \
        _parse_smplx_pose(pose_params, num_pose_values)

    betas = shape_params.float().reshape(1, -1)

    output = body_model(
        global_orient=global_orient,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        betas=betas,
        return_verts=True,
    )

    vertices = output.vertices  # (1, V, 3)

    # Remove batch dimension
    if vertices.shape[0] == 1:
        vertices = vertices[0]  # (V, 3)
    else:
        raise NotImplementedError("Batch mesh not supported yet")

    if transform is not None:
        vertices = apply_translation(vertices, transform)

    vertices_mi = mi.TensorXf(vertices.cpu().detach().numpy())

    if not need_face:
        return vertices_mi

    faces = body_model.faces_tensor.long()  # (F, 3)
    faces_mi = mi.TensorXf(faces.cpu().numpy().astype(np.float32))
    return vertices_mi, faces_mi


# Backward-compatible alias
call_smpl_layer = call_smplx_layer
