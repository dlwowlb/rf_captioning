

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import io
import cv2
from tqdm import tqdm

from genesis.raytracing.radar import Radar
from genesis.visualization.pointcloud import PointCloudProcessCFG, frame2pointcloud,rangeFFT,dopplerFFT,process_pc

# Try to import SMPL_Layer, but don't fail if not available
try:
    from smplpytorch.pytorch.smpl_layer import SMPL_Layer
    _HAS_SMPL_LAYER = True
except ImportError:
    _HAS_SMPL_LAYER = False


# SMPL-H 52-joint parent indices for skeleton drawing
# child -> parent mapping (same as object_diff.SMPLH_PARENTS)
SMPLH_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    # Left hand
    20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
    # Right hand
    21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50,
]

# SMPL 24-joint kintree for skeleton drawing (child, parent pairs)
SMPL24_KINTREE = [
    (1, 0), (2, 0), (3, 0), (4, 1), (5, 2), (6, 3),
    (7, 4), (8, 5), (9, 6), (10, 7), (11, 8), (12, 9),
    (13, 9), (14, 9), (15, 12), (16, 13), (17, 14),
    (18, 16), (19, 17), (20, 18), (21, 19), (22, 20), (23, 21),
]


def display_smpl(
        model_info,
        model_faces=None,
        with_joints=False,
        kintree_table=None,
        ax=None,
        batch_idx=0,
        translation=None,
        ):
    """
    Displays mesh batch_idx in batch of model_info, model_info as returned by
    generate_random_model
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts, joints = model_info['verts'][batch_idx], model_info['joints'][
        batch_idx]
    if translation is not None:
        verts += translation
        joints += translation

    if model_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.2)
    else:
        mesh = Poly3DCollection(verts[model_faces], alpha=0.2)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    if with_joints:
        draw_skeleton(joints, kintree_table=kintree_table, ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.5, 2)
    ax.set_zlim(-1, 3)
    ax.view_init(azim=-90, elev=100)
    ax.view_init(azim=30, elev=30, roll = 105)
    ax.set_title('SMPL model', fontsize=20)
    return ax


def draw_skeleton(joints3D, kintree_table, ax=None, with_numbers=False):
    if ax is None:
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = ax

    colors = []
    left_right_mid = ['r', 'g', 'b']
    kintree_colors = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1]
    for c in kintree_colors:
        colors += left_right_mid[c]
    # For each 24 joint
    for i in range(1, kintree_table.shape[1]):
        j1 = kintree_table[0][i]
        j2 = kintree_table[1][i]
        ax.plot([joints3D[j1, 0], joints3D[j2, 0]],
                [joints3D[j1, 1], joints3D[j2, 1]],
                [joints3D[j1, 2], joints3D[j2, 2]],
                color=colors[i], linestyle='-', linewidth=2, marker='o', markersize=5)
        if with_numbers:
            ax.text(joints3D[j2, 0], joints3D[j2, 1], joints3D[j2, 2], j2)
    return ax


def draw_hand_skeleton(joints3D, ax, hand='left', color=None):
    """Draw hand skeleton on 3D axis.

    Args:
        joints3D: (52, 3) all joint positions (needs joints 20-36 for left, 21+37-51 for right)
        ax: matplotlib 3D axis
        hand: 'left' or 'right'
        color: color for the hand skeleton lines
    """
    if hand == 'left':
        # Left hand: joints 22-36, parent = L_Wrist (20)
        joint_range = range(22, 37)
        base_color = color or 'orange'
    else:
        # Right hand: joints 37-51, parent = R_Wrist (21)
        joint_range = range(37, 52)
        base_color = color or 'cyan'

    for jidx in joint_range:
        if jidx >= joints3D.shape[0]:
            break
        parent = SMPLH_PARENTS[jidx]
        if parent < 0 or parent >= joints3D.shape[0]:
            continue
        ax.plot(
            [joints3D[parent, 0], joints3D[jidx, 0]],
            [joints3D[parent, 1], joints3D[jidx, 1]],
            [joints3D[parent, 2], joints3D[jidx, 2]],
            color=base_color, linestyle='-', linewidth=1.5, marker='o', markersize=3,
        )


def draw_skeleton_smplh(joints3D, ax=None, with_hands=True):
    """Draw full SMPL-H 52-joint skeleton including hands.

    Args:
        joints3D: (J, 3) joint positions, J can be 22, 24, or 52
        ax: matplotlib 3D axis
        with_hands: whether to draw hand joints (if available)
    """
    if ax is None:
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111, projection='3d')

    num_joints = joints3D.shape[0]

    # Draw body skeleton (joints 0-21 or 0-23)
    left_right_mid = ['r', 'g', 'b']
    body_colors = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1]

    body_max = min(num_joints, 24)
    for child, parent in SMPL24_KINTREE:
        if child >= body_max or parent >= body_max:
            continue
        c = left_right_mid[body_colors[child]] if child < len(body_colors) else 'b'
        ax.plot(
            [joints3D[parent, 0], joints3D[child, 0]],
            [joints3D[parent, 1], joints3D[child, 1]],
            [joints3D[parent, 2], joints3D[child, 2]],
            color=c, linestyle='-', linewidth=2, marker='o', markersize=4,
        )

    # Draw hand skeletons if available
    if with_hands and num_joints > 22:
        draw_hand_skeleton(joints3D, ax, hand='left', color='orange')
        draw_hand_skeleton(joints3D, ax, hand='right', color='cyan')

    return ax


def draw_smpl_on_axis(pose, shape, translation=None, ax=None, keypoints3d=None, smpl_model_root=None):
    """Draw SMPL body on axis.

    If keypoints3d is provided, draws the skeleton directly from the joint positions
    (no SMPL_Layer needed). Otherwise falls back to SMPL_Layer if available.

    Args:
        pose: (72,) or (156,) axis-angle pose
        shape: (10,) body shape
        translation: (3,) translation offset
        ax: matplotlib 3D axis
        keypoints3d: (J, 3) pre-computed joint positions (22, 24, or 52 joints)
        smpl_model_root: path to SMPL model files (for SMPL_Layer fallback)
    """
    if keypoints3d is not None:
        # Use pre-computed keypoints3d directly
        joints = keypoints3d.copy()
        if translation is not None:
            joints = joints + translation

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        has_hands = joints.shape[0] > 24
        draw_skeleton_smplh(joints, ax=ax, with_hands=has_hands)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.5, 2)
        ax.set_zlim(-1, 3)
        ax.view_init(azim=30, elev=30, roll=105)
        ax.set_title('SMPL-H skeleton' if has_hands else 'SMPL skeleton', fontsize=20)
        return

    # Fallback: use SMPL_Layer to compute vertices and joints
    if not _HAS_SMPL_LAYER:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.set_title('SMPL (no model)', fontsize=20)
        ax.text(0, 1, 0, 'SMPL_Layer not available', fontsize=10, color='red')
        return

    pose = torch.tensor(pose).unsqueeze(0)
    shape = torch.tensor(shape).unsqueeze(0)
    model_root = smpl_model_root or '../models/smpl_models'
    smpl_layer = SMPL_Layer(center_idx=0, gender='male', model_root=model_root)
    verts, Jtr = smpl_layer(pose.float(), th_betas=shape.float())

    display_smpl(
        {'verts': verts.cpu().detach(),
         'joints': Jtr.cpu().detach()},
        model_faces=smpl_layer.th_faces,
        with_joints=True,
        kintree_table=smpl_layer.kintree_table, translation=translation, ax=ax)


# Plotting Pointclouds
def draw_poinclouds_on_axis(pc,ax, tx,rx,elev,azim,title):
    pc = np.transpose(pc)
    ax.scatter(-pc[0], pc[1], pc[2], c=pc[4], cmap=plt.hot())
    if tx is not None:
        ax.scatter(tx[:,0], tx[:,2], tx[:,1], c="green", s= 50, marker =',', cmap=plt.hot())
    if rx is not None:
        ax.scatter(rx[:,0], rx[:,2], rx[:,1], c="orange", s= 50, marker =',', cmap=plt.hot())
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0, 6)
    ax.set_zlim(-0.5, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=20)

def draw_doppler_on_axis(radar_frame,pointcloud_cfg, ax):
    range_fft = rangeFFT(radar_frame,pointcloud_cfg.frameConfig)
    doppler_fft = dopplerFFT(range_fft,pointcloud_cfg.frameConfig)
    dopplerResultSumAllAntenna = np.sum(doppler_fft, axis=(0,1))
    ax.imshow(np.abs(dopplerResultSumAllAntenna))
    ax.set_title("Doppler FFT", fontsize=20)

def draw_combined(i, pointcloud_cfg, radar_frames, pointclouds, smpl_data):
    smpl_frame_id = i               # 30FPS
    radar_frame_id = int(i/3)       # 10FPS

    poses = smpl_data["pose"]
    shape = smpl_data['shape']
    root_translation = smpl_data['root_translation']

    # Check for SMPL-H keypoints3d
    has_keypoints3d = 'keypoints3d' in smpl_data and smpl_data['keypoints3d'] is not None
    if has_keypoints3d:
        kp3d = smpl_data['keypoints3d']
        # keypoints3d already includes translation, so don't add it again
        frame_kp3d = kp3d[smpl_frame_id]
    else:
        frame_kp3d = None

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(131, projection='3d')
    draw_smpl_on_axis(
        poses[smpl_frame_id], shape,
        translation=root_translation[smpl_frame_id],
        ax=ax1,
        keypoints3d=frame_kp3d,
    )

    ax2 = fig.add_subplot(132, projection='3d')
    draw_poinclouds_on_axis(pointclouds[radar_frame_id],ax2, None,None,30,-30,"Point clouds")

    ax3 = fig.add_subplot(133)
    draw_doppler_on_axis(radar_frames[radar_frame_id],pointcloud_cfg, ax3)

    plt.tight_layout()
    fig.canvas.draw()
    data = np.asarray(fig.canvas.buffer_rgba())
    data = data[:, :, :3] # RGBA -> RGB

    plt.close(fig)
    return data


def save_video(radar_cfg_file, radar_frames_file, smpl_data_file, output_file):
    radar = Radar(radar_cfg_file)
    pointcloud_cfg = PointCloudProcessCFG(radar)
    radar_frames = np.load(radar_frames_file)
    smpl_data = dict(np.load(smpl_data_file, allow_pickle=True))

    # Process the pointclouds
    pointclouds = []
    for frame in radar_frames:
        pc = process_pc(pointcloud_cfg, frame)
        pointclouds.append(pc)

    # Write the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_filename = output_file
    out = cv2.VideoWriter(video_filename, fourcc, 30, (1200, 600))
    for i in tqdm(range(smpl_data["pose"].shape[0]-2)):
        frame = draw_combined(i,pointcloud_cfg,radar_frames,pointclouds,smpl_data)
        rgb_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(rgb_data)
    out.release()
