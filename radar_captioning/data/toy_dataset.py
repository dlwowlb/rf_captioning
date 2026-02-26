"""
Toy Dataset Generator for Radar Captioning.

Generates synthetic radar heatmaps (Range-Doppler, Range-Angle, Doppler-Angle)
paired with text descriptions and physical parameters.

Uses SMPL-H 52-joint skeleton (22 body + 15 left hand + 15 right hand) to
simulate realistic motion patterns, then converts to radar signatures.

Each sample contains:
  - rd_heatmap: (T, H, W) Range-Doppler heatmaps
  - ra_heatmap: (T, H, W) Range-Angle heatmaps
  - da_heatmap: (T, H, W) Doppler-Angle heatmaps
  - text: ground truth description
  - physical_params: dict with distance, speed, angle
  - action_label: categorical action label
  - curriculum_phase: 'basic', 'composition', or 'occlusion'
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ============================================================================
# SMPL-H Joint Definitions (52 joints)
# ============================================================================

SMPLH_JOINT_NAMES = [
    # 22 body joints
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    # 15 left hand joints
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    # 15 right hand joints
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
]

# Parent joint indices for SMPL-H kinematic tree
SMPLH_PARENTS = [
    -1,  # 0 pelvis (root)
    0, 0, 0, 1, 2,           # 1-5: hips, spine1, knees
    3, 4, 5, 6, 7, 8,       # 6-11: spine2, ankles, spine3, feet
    9, 9, 9, 12, 13, 14,    # 12-17: neck, collars, head, shoulders
    16, 17, 18, 19,          # 18-21: elbows, wrists
    # Left hand (parent = left_wrist = 20)
    20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
    # Right hand (parent = right_wrist = 21)
    21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50,
]


# ============================================================================
# Action Definitions with Physical Parameters
# ============================================================================

BASIC_ACTIONS = {
    "walking_forward": {
        "text_templates": [
            "A person walks forward at {speed:.1f} m/s, located {dist:.1f}m away at {angle:.0f} degrees.",
            "Someone is walking straight ahead at a distance of {dist:.1f}m, moving at {speed:.1f} m/s, angle {angle:.0f} degrees.",
        ],
        "speed_range": (0.8, 1.5),  # m/s
        "dist_range": (2.0, 6.0),   # m
        "angle_range": (-10, 10),    # degrees
        "doppler_sign": 1,           # approaching
        "body_motion": "linear_forward",
    },
    "walking_backward": {
        "text_templates": [
            "A person walks backward at {speed:.1f} m/s, located {dist:.1f}m away at {angle:.0f} degrees.",
        ],
        "speed_range": (0.5, 1.0),
        "dist_range": (2.0, 6.0),
        "angle_range": (-10, 10),
        "doppler_sign": -1,          # receding
        "body_motion": "linear_backward",
    },
    "sitting_down": {
        "text_templates": [
            "A person sits down slowly at {dist:.1f}m distance, {angle:.0f} degrees, speed {speed:.1f} m/s.",
            "Someone is sitting down at a distance of {dist:.1f}m, angle {angle:.0f} degrees, vertical speed {speed:.1f} m/s.",
        ],
        "speed_range": (0.2, 0.5),
        "dist_range": (1.5, 4.0),
        "angle_range": (-20, 20),
        "doppler_sign": -1,
        "body_motion": "vertical_down",
    },
    "standing_up": {
        "text_templates": [
            "A person stands up at {dist:.1f}m away, {angle:.0f} degrees, speed {speed:.1f} m/s.",
        ],
        "speed_range": (0.3, 0.6),
        "dist_range": (1.5, 4.0),
        "angle_range": (-20, 20),
        "doppler_sign": 1,
        "body_motion": "vertical_up",
    },
    "standing_still": {
        "text_templates": [
            "A person stands still at {dist:.1f}m away, {angle:.0f} degrees, near-zero speed {speed:.2f} m/s.",
        ],
        "speed_range": (0.0, 0.05),
        "dist_range": (1.5, 5.0),
        "angle_range": (-30, 30),
        "doppler_sign": 0,
        "body_motion": "static",
    },
}

COMPOSITION_ACTIONS = {
    "walking_waving": {
        "text_templates": [
            "A person walks forward at {speed:.1f} m/s while waving a hand, {dist:.1f}m away at {angle:.0f} degrees.",
        ],
        "speed_range": (0.8, 1.3),
        "dist_range": (2.0, 6.0),
        "angle_range": (-15, 15),
        "doppler_sign": 1,
        "body_motion": "walk_and_wave",
    },
    "walking_phone": {
        "text_templates": [
            "A person walks at {speed:.1f} m/s while talking on a phone, distance {dist:.1f}m, angle {angle:.0f} degrees.",
        ],
        "speed_range": (0.6, 1.2),
        "dist_range": (2.0, 6.0),
        "angle_range": (-15, 15),
        "doppler_sign": 1,
        "body_motion": "walk_and_phone",
    },
    "jumping_in_place": {
        "text_templates": [
            "A person jumps in place at {dist:.1f}m away, {angle:.0f} degrees, peak speed {speed:.1f} m/s.",
        ],
        "speed_range": (1.0, 2.0),
        "dist_range": (2.0, 5.0),
        "angle_range": (-10, 10),
        "doppler_sign": 0,  # alternating
        "body_motion": "jump_in_place",
    },
    "stretching_arms": {
        "text_templates": [
            "A person stretches arms overhead at {dist:.1f}m, {angle:.0f} degrees, arm speed {speed:.1f} m/s.",
        ],
        "speed_range": (0.3, 0.8),
        "dist_range": (2.0, 5.0),
        "angle_range": (-20, 20),
        "doppler_sign": 0,
        "body_motion": "stretch_arms",
    },
}


def _generate_smplh_motion(
    action_type: str,
    num_frames: int,
    speed: float,
    distance: float,
    angle_deg: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Generate synthetic SMPL-H joint positions for a given action.

    Returns:
        joint_positions: (num_frames, 52, 3) - xyz positions for all 52 joints
    """
    n_joints = 52
    positions = np.zeros((num_frames, n_joints, 3))
    t = np.linspace(0, 1, num_frames)
    angle_rad = np.radians(angle_deg)

    # Base position: place person at (distance*sin(angle), 0, distance*cos(angle))
    base_x = distance * np.sin(angle_rad)
    base_y = 0.0  # ground level
    base_z = distance * np.cos(angle_rad)

    # Generate body joint offsets from a T-pose template
    # Approximate SMPL-H rest pose heights (relative to pelvis)
    rest_offsets = np.zeros((n_joints, 3))
    # Body joints approximate positions (from pelvis at origin)
    rest_offsets[0] = [0, 0.9, 0]       # pelvis
    rest_offsets[1] = [-0.1, 0.85, 0]   # left_hip
    rest_offsets[2] = [0.1, 0.85, 0]    # right_hip
    rest_offsets[3] = [0, 1.05, 0]      # spine1
    rest_offsets[4] = [-0.1, 0.45, 0]   # left_knee
    rest_offsets[5] = [0.1, 0.45, 0]    # right_knee
    rest_offsets[6] = [0, 1.2, 0]       # spine2
    rest_offsets[7] = [-0.1, 0.05, 0]   # left_ankle
    rest_offsets[8] = [0.1, 0.05, 0]    # right_ankle
    rest_offsets[9] = [0, 1.35, 0]      # spine3
    rest_offsets[10] = [-0.1, 0.0, 0.05]  # left_foot
    rest_offsets[11] = [0.1, 0.0, 0.05]   # right_foot
    rest_offsets[12] = [0, 1.55, 0]      # neck
    rest_offsets[13] = [-0.15, 1.5, 0]   # left_collar
    rest_offsets[14] = [0.15, 1.5, 0]    # right_collar
    rest_offsets[15] = [0, 1.7, 0]       # head
    rest_offsets[16] = [-0.25, 1.45, 0]  # left_shoulder
    rest_offsets[17] = [0.25, 1.45, 0]   # right_shoulder
    rest_offsets[18] = [-0.45, 1.35, 0]  # left_elbow
    rest_offsets[19] = [0.45, 1.35, 0]   # right_elbow
    rest_offsets[20] = [-0.55, 1.2, 0]   # left_wrist
    rest_offsets[21] = [0.55, 1.2, 0]    # right_wrist

    # Left hand fingers (relative to left_wrist)
    for i in range(15):
        finger_idx = 22 + i
        finger_group = i // 3  # 0-4 for index, middle, pinky, ring, thumb
        finger_joint = i % 3   # 0-2 for base, mid, tip
        x_off = -0.02 * (finger_group - 2)
        y_off = -0.02 * (finger_joint + 1)
        rest_offsets[finger_idx] = rest_offsets[20] + [x_off, y_off, 0.01 * (finger_joint + 1)]

    # Right hand fingers (relative to right_wrist)
    for i in range(15):
        finger_idx = 37 + i
        finger_group = i // 3
        finger_joint = i % 3
        x_off = 0.02 * (finger_group - 2)
        y_off = -0.02 * (finger_joint + 1)
        rest_offsets[finger_idx] = rest_offsets[21] + [x_off, y_off, 0.01 * (finger_joint + 1)]

    # Apply action-specific motion
    for f in range(num_frames):
        # Start from rest pose + base position
        positions[f] = rest_offsets.copy()
        positions[f, :, 0] += base_x
        positions[f, :, 2] += base_z

        if action_type == "linear_forward":
            # Walking forward: translate along z, leg swing
            dz = speed * t[f]
            positions[f, :, 2] -= dz  # approaching radar
            # Leg swing (pendulum)
            swing = 0.15 * np.sin(2 * np.pi * 2 * t[f])
            positions[f, 4, 2] += swing   # left knee forward
            positions[f, 5, 2] -= swing   # right knee backward
            positions[f, 7, 2] += swing * 0.8
            positions[f, 8, 2] -= swing * 0.8
            # Arm swing (opposite to legs)
            positions[f, 18, 2] -= swing * 0.3
            positions[f, 19, 2] += swing * 0.3
            # Finger micro-motion during walk
            for j in range(22, 37):
                positions[f, j, 1] += rng.normal(0, 0.003)
            for j in range(37, 52):
                positions[f, j, 1] += rng.normal(0, 0.003)

        elif action_type == "linear_backward":
            dz = speed * t[f]
            positions[f, :, 2] += dz  # receding from radar
            swing = 0.12 * np.sin(2 * np.pi * 2 * t[f])
            positions[f, 4, 2] -= swing
            positions[f, 5, 2] += swing

        elif action_type == "vertical_down":
            # Sitting: pelvis moves down, knees bend
            progress = t[f]
            sit_amount = 0.4 * progress
            positions[f, 0, 1] -= sit_amount   # pelvis down
            for j in [3, 6, 9, 12, 13, 14, 15, 16, 17]:
                positions[f, j, 1] -= sit_amount
            # Knees move forward as person sits
            positions[f, 4, 2] += 0.2 * progress
            positions[f, 5, 2] += 0.2 * progress
            positions[f, 4, 1] -= sit_amount * 0.5
            positions[f, 5, 1] -= sit_amount * 0.5
            # Hands move to lap
            positions[f, 20, 1] -= sit_amount * 0.8
            positions[f, 21, 1] -= sit_amount * 0.8
            for j in range(22, 52):
                positions[f, j, 1] -= sit_amount * 0.8

        elif action_type == "vertical_up":
            progress = t[f]
            stand_amount = 0.4 * progress
            positions[f, 0, 1] += stand_amount
            for j in [3, 6, 9, 12, 13, 14, 15, 16, 17]:
                positions[f, j, 1] += stand_amount
            positions[f, 4, 2] -= 0.2 * progress
            positions[f, 5, 2] -= 0.2 * progress

        elif action_type == "static":
            # Small breathing motion
            breath = 0.01 * np.sin(2 * np.pi * 0.3 * t[f])
            positions[f, 9, 1] += breath
            positions[f, 6, 1] += breath * 0.5

        elif action_type == "walk_and_wave":
            # Walking + right hand waving
            dz = speed * t[f]
            positions[f, :, 2] -= dz
            swing = 0.12 * np.sin(2 * np.pi * 2 * t[f])
            positions[f, 4, 2] += swing
            positions[f, 5, 2] -= swing
            # Right arm wave
            wave = 0.3 * np.sin(2 * np.pi * 3 * t[f])
            positions[f, 19, 1] += 0.3 + wave * 0.2  # elbow up
            positions[f, 21, 1] += 0.5 + wave * 0.3   # wrist wave
            # Right hand fingers spread during wave
            for j in range(37, 52):
                positions[f, j, 1] += 0.5 + wave * 0.3
                positions[f, j, 0] += 0.02 * ((j - 37) % 3)  # spread fingers

        elif action_type == "walk_and_phone":
            dz = speed * t[f]
            positions[f, :, 2] -= dz
            swing = 0.1 * np.sin(2 * np.pi * 2 * t[f])
            positions[f, 4, 2] += swing
            positions[f, 5, 2] -= swing
            # Right hand to ear (phone)
            positions[f, 19, 1] += 0.2
            positions[f, 21, 0] += 0.1
            positions[f, 21, 1] += 0.45
            # Fingers curl around phone
            for j in range(37, 52):
                positions[f, j, 0] += 0.08
                positions[f, j, 1] += 0.45
                positions[f, j, 2] += 0.02 * ((j - 37) % 3)

        elif action_type == "jump_in_place":
            # Vertical jump cycle
            phase = t[f] * 3  # 3 jumps in duration
            jump_h = max(0, 0.3 * np.sin(np.pi * (phase % 1)))
            positions[f, :, 1] += jump_h
            # Arms go up during jump
            positions[f, 18, 1] += jump_h * 0.5
            positions[f, 19, 1] += jump_h * 0.5
            positions[f, 20, 1] += jump_h * 0.7
            positions[f, 21, 1] += jump_h * 0.7

        elif action_type == "stretch_arms":
            progress = min(t[f] * 2, 1.0)
            # Arms go overhead
            positions[f, 16, 1] += 0.3 * progress  # left shoulder
            positions[f, 17, 1] += 0.3 * progress  # right shoulder
            positions[f, 18, 1] += 0.4 * progress  # left elbow
            positions[f, 19, 1] += 0.4 * progress
            positions[f, 20, 1] += 0.6 * progress  # left wrist
            positions[f, 21, 1] += 0.6 * progress
            positions[f, 20, 0] += 0.1 * progress  # spread
            positions[f, 21, 0] -= 0.1 * progress
            # Fingers extend
            for j in range(22, 52):
                positions[f, j, 1] += 0.6 * progress

        # Add small noise to all joints
        positions[f] += rng.normal(0, 0.002, (n_joints, 3))

    return positions


def _joints_to_radar_heatmaps(
    joint_positions: np.ndarray,
    heatmap_size: int = 64,
    max_range: float = 10.0,
    max_doppler: float = 5.0,
    max_angle: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert SMPL-H joint positions to radar heatmaps.

    Computes Range, Doppler, and Angle from joint positions and velocities,
    then creates 2D heatmap projections.

    Args:
        joint_positions: (T, 52, 3) joint xyz positions
        heatmap_size: output heatmap H=W

    Returns:
        rd_maps: (T, H, W) Range-Doppler heatmaps
        ra_maps: (T, H, W) Range-Angle heatmaps
        da_maps: (T, H, W) Doppler-Angle heatmaps
    """
    T, n_joints, _ = joint_positions.shape
    H = W = heatmap_size

    rd_maps = np.zeros((T, H, W))
    ra_maps = np.zeros((T, H, W))
    da_maps = np.zeros((T, H, W))

    for frame_idx in range(T):
        pos = joint_positions[frame_idx]  # (52, 3)

        # Compute range (distance from radar at origin)
        ranges = np.sqrt(np.sum(pos ** 2, axis=1))  # (52,)

        # Compute angle (azimuth from z-axis)
        angles = np.degrees(np.arctan2(pos[:, 0], pos[:, 2]))  # (52,)

        # Compute Doppler velocity (radial component)
        if frame_idx > 0:
            vel = (joint_positions[frame_idx] - joint_positions[frame_idx - 1]) * 30  # 30fps
            # Radial velocity = dot(vel, unit_range_vector)
            range_unit = pos / (ranges[:, None] + 1e-8)
            doppler = np.sum(vel * range_unit, axis=1)  # (52,) radial velocity
        else:
            doppler = np.zeros(n_joints)

        # Map to heatmap bins
        range_bins = np.clip((ranges / max_range * H), 0, H - 1).astype(int)
        doppler_bins = np.clip(((doppler / max_doppler + 1) / 2 * W), 0, W - 1).astype(int)
        angle_bins = np.clip(((angles / max_angle + 1) / 2 * W), 0, W - 1).astype(int)

        # Joint intensity (larger joints = stronger reflection)
        intensities = np.ones(n_joints)
        intensities[:22] = 1.0       # body joints
        intensities[22:37] = 0.3     # left hand (smaller RCS)
        intensities[37:52] = 0.3     # right hand

        # Fill heatmaps with Gaussian spread
        sigma = 1.5
        for j in range(n_joints):
            w = intensities[j]
            # Range-Doppler
            _add_gaussian(rd_maps[frame_idx], range_bins[j], doppler_bins[j], w, sigma)
            # Range-Angle
            _add_gaussian(ra_maps[frame_idx], range_bins[j], angle_bins[j], w, sigma)
            # Doppler-Angle
            _add_gaussian(da_maps[frame_idx], doppler_bins[j], angle_bins[j], w, sigma)

    # Normalize to [0, 1]
    for maps in [rd_maps, ra_maps, da_maps]:
        vmax = maps.max()
        if vmax > 0:
            maps /= vmax

    return rd_maps, ra_maps, da_maps


def _add_gaussian(heatmap: np.ndarray, cy: int, cx: int, weight: float, sigma: float):
    """Add a weighted Gaussian blob to a 2D heatmap."""
    H, W = heatmap.shape
    r = int(3 * sigma)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            y, x = cy + dy, cx + dx
            if 0 <= y < H and 0 <= x < W:
                g = weight * np.exp(-(dy ** 2 + dx ** 2) / (2 * sigma ** 2))
                heatmap[y, x] += g


def _apply_occlusion(
    rd: np.ndarray, ra: np.ndarray, da: np.ndarray,
    rng: np.random.RandomState,
    occlusion_strength: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply synthetic wall/occlusion effects to heatmaps."""
    T, H, W = rd.shape

    # Add random noise bands (simulating wall reflections)
    noise_range_start = rng.randint(0, H // 2)
    noise_range_end = noise_range_start + rng.randint(3, 8)
    noise_level = rng.uniform(0.1, 0.3)

    for f in range(T):
        # Wall reflection noise
        wall_noise = rng.uniform(0, noise_level, (min(noise_range_end, H) - noise_range_start, W))
        rd[f, noise_range_start:min(noise_range_end, H)] += wall_noise
        ra[f, noise_range_start:min(noise_range_end, H)] += wall_noise[:, :W]

        # Attenuation of signal (simulating through-wall)
        attenuation = rng.uniform(0.3, 0.7)
        rd[f] *= attenuation
        ra[f] *= attenuation
        da[f] *= attenuation

        # Random dropout patches (simulating multipath)
        if rng.random() < occlusion_strength:
            patch_y = rng.randint(0, H - 8)
            patch_x = rng.randint(0, W - 8)
            patch_h = rng.randint(4, 12)
            patch_w = rng.randint(4, 12)
            rd[f, patch_y:min(patch_y + patch_h, H), patch_x:min(patch_x + patch_w, W)] *= 0.1
            ra[f, patch_y:min(patch_y + patch_h, H), patch_x:min(patch_x + patch_w, W)] *= 0.1

    # Re-normalize
    for maps in [rd, ra, da]:
        vmax = maps.max()
        if vmax > 0:
            maps /= vmax

    return rd, ra, da


def generate_toy_dataset(
    num_samples: int = 300,
    num_frames: int = 30,
    heatmap_size: int = 64,
    output_dir: str = "data/toy_dataset",
    seed: int = 42,
) -> List[Dict]:
    """
    Generate complete toy dataset with all curriculum phases.

    Distribution:
      - 40% basic actions
      - 35% composition actions
      - 25% occlusion variants

    Returns:
        List of sample dictionaries
    """
    rng = np.random.RandomState(seed)
    os.makedirs(output_dir, exist_ok=True)

    n_basic = int(num_samples * 0.40)
    n_composition = int(num_samples * 0.35)
    n_occlusion = num_samples - n_basic - n_composition

    samples = []
    sample_id = 0

    all_actions = {
        "basic": list(BASIC_ACTIONS.items()),
        "composition": list(COMPOSITION_ACTIONS.items()),
    }

    # Phase 1: Basic actions
    for i in range(n_basic):
        action_name, action_cfg = all_actions["basic"][i % len(all_actions["basic"])]
        sample = _generate_single_sample(
            sample_id, action_name, action_cfg, "basic",
            num_frames, heatmap_size, rng, occlude=False,
        )
        samples.append(sample)
        sample_id += 1

    # Phase 2: Composition actions
    for i in range(n_composition):
        action_name, action_cfg = all_actions["composition"][i % len(all_actions["composition"])]
        sample = _generate_single_sample(
            sample_id, action_name, action_cfg, "composition",
            num_frames, heatmap_size, rng, occlude=False,
        )
        samples.append(sample)
        sample_id += 1

    # Phase 3: Occlusion (70% occluded from basic/composition + 30% clean)
    n_occluded = int(n_occlusion * 0.7)
    n_clean = n_occlusion - n_occluded

    all_action_list = all_actions["basic"] + all_actions["composition"]
    for i in range(n_occluded):
        action_name, action_cfg = all_action_list[i % len(all_action_list)]
        sample = _generate_single_sample(
            sample_id, action_name, action_cfg, "occlusion",
            num_frames, heatmap_size, rng, occlude=True,
        )
        samples.append(sample)
        sample_id += 1

    for i in range(n_clean):
        action_name, action_cfg = all_action_list[i % len(all_action_list)]
        sample = _generate_single_sample(
            sample_id, action_name, action_cfg, "occlusion",
            num_frames, heatmap_size, rng, occlude=False,
        )
        samples.append(sample)
        sample_id += 1

    # Save to disk
    _save_dataset(samples, output_dir)
    print(f"Generated {len(samples)} samples -> {output_dir}")
    print(f"  Basic: {n_basic}, Composition: {n_composition}, "
          f"Occlusion: {n_occlusion} ({n_occluded} occluded + {n_clean} clean)")

    return samples


def _generate_single_sample(
    sample_id: int,
    action_name: str,
    action_cfg: dict,
    curriculum_phase: str,
    num_frames: int,
    heatmap_size: int,
    rng: np.random.RandomState,
    occlude: bool = False,
) -> Dict:
    """Generate a single sample with radar heatmaps and text."""
    # Sample physical parameters
    speed = rng.uniform(*action_cfg["speed_range"])
    dist = rng.uniform(*action_cfg["dist_range"])
    angle = rng.uniform(*action_cfg["angle_range"])

    # Generate SMPL-H motion
    joint_positions = _generate_smplh_motion(
        action_cfg["body_motion"], num_frames, speed, dist, angle, rng
    )

    # Convert to radar heatmaps
    rd, ra, da = _joints_to_radar_heatmaps(
        joint_positions, heatmap_size,
        max_range=10.0, max_doppler=5.0, max_angle=60.0,
    )

    # Apply occlusion if needed
    if occlude:
        rd, ra, da = _apply_occlusion(rd, ra, da, rng)

    # Generate text description
    template = rng.choice(action_cfg["text_templates"])
    text = template.format(speed=speed, dist=dist, angle=angle)

    # Compute actual Doppler sign for JSON physical slots
    doppler_sign = action_cfg["doppler_sign"]
    if doppler_sign == 0:
        doppler_desc = "near-zero"
    elif doppler_sign > 0:
        doppler_desc = "positive (approaching)"
    else:
        doppler_desc = "negative (receding)"

    # Physical slots JSON
    physical_slots = {
        "distance_m": round(float(dist), 2),
        "speed_ms": round(float(speed), 2),
        "angle_deg": round(float(angle), 1),
        "doppler_sign": doppler_desc,
        "action": action_name,
    }

    return {
        "id": sample_id,
        "rd_heatmap": rd.astype(np.float32),      # (T, H, W)
        "ra_heatmap": ra.astype(np.float32),       # (T, H, W)
        "da_heatmap": da.astype(np.float32),       # (T, H, W)
        "joint_positions": joint_positions.astype(np.float32),  # (T, 52, 3)
        "text": text,
        "physical_slots": physical_slots,
        "action_label": action_name,
        "curriculum_phase": curriculum_phase,
    }


def _save_dataset(samples: List[Dict], output_dir: str):
    """Save dataset to disk as numpy arrays and JSON metadata."""
    os.makedirs(output_dir, exist_ok=True)

    metadata = []
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    for s in samples:
        sid = s["id"]
        # Save heatmaps as compressed numpy
        np.savez_compressed(
            os.path.join(heatmap_dir, f"sample_{sid:04d}.npz"),
            rd=s["rd_heatmap"],
            ra=s["ra_heatmap"],
            da=s["da_heatmap"],
            joints=s["joint_positions"],
        )
        metadata.append({
            "id": sid,
            "text": s["text"],
            "physical_slots": s["physical_slots"],
            "action_label": s["action_label"],
            "curriculum_phase": s["curriculum_phase"],
        })

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


class RadarCaptionDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for radar captioning."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        train_ratio: float = 0.8,
        curriculum_phase: Optional[str] = None,
        num_frames: int = 30,
    ):
        self.data_dir = data_dir
        self.num_frames = num_frames

        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            all_metadata = json.load(f)

        # Filter by curriculum phase if specified
        if curriculum_phase is not None:
            all_metadata = [m for m in all_metadata if m["curriculum_phase"] == curriculum_phase]

        # Split
        n = len(all_metadata)
        n_train = int(n * train_ratio)
        if split == "train":
            self.metadata = all_metadata[:n_train]
        else:
            self.metadata = all_metadata[n_train:]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        sid = meta["id"]

        # Load heatmaps
        data = np.load(
            os.path.join(self.data_dir, "heatmaps", f"sample_{sid:04d}.npz")
        )

        T = self.num_frames
        rd = torch.from_numpy(data["rd"][:T])  # (T, H, W)
        ra = torch.from_numpy(data["ra"][:T])
        da = torch.from_numpy(data["da"][:T])

        # Pad if needed
        if rd.shape[0] < T:
            pad = T - rd.shape[0]
            rd = torch.nn.functional.pad(rd, (0, 0, 0, 0, 0, pad))
            ra = torch.nn.functional.pad(ra, (0, 0, 0, 0, 0, pad))
            da = torch.nn.functional.pad(da, (0, 0, 0, 0, 0, pad))

        return {
            "rd": rd,               # (T, H, W)
            "ra": ra,               # (T, H, W)
            "da": da,               # (T, H, W)
            "text": meta["text"],
            "physical_slots": json.dumps(meta["physical_slots"]),
            "action_label": meta["action_label"],
            "sample_id": sid,
        }


if __name__ == "__main__":
    # Quick test
    samples = generate_toy_dataset(
        num_samples=10, num_frames=30, heatmap_size=64,
        output_dir="/tmp/test_toy_dataset", seed=42,
    )
    print(f"Sample 0 text: {samples[0]['text']}")
    print(f"Sample 0 RD shape: {samples[0]['rd_heatmap'].shape}")
    print(f"Sample 0 joints shape: {samples[0]['joint_positions'].shape}")
    print(f"Sample 0 physical: {samples[0]['physical_slots']}")
