import drjit as dr
import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T
import numpy as np
from . import smpl
import torch
from tqdm import tqdm
import os
import tempfile

mi.set_variant('cuda_ad_rgb')
torch.set_default_device('cuda')


def _ensure_ply_mesh(body_model, ply_path):
    """
    Generate a PLY mesh file from the SMPL-X body model's T-pose.

    The pathtracer scene requires a .ply file for Mitsuba to load.
    We generate it from the SMPL-X model's rest-pose vertices and faces.

    Args:
        body_model: SMPL-X model instance
        ply_path: Path to write the .ply file
    """
    if os.path.exists(ply_path):
        return

    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    # Get T-pose vertices
    output = body_model(
        return_verts=True,
    )
    vertices = output.vertices[0].cpu().detach().numpy()  # (V, 3)
    faces = body_model.faces_tensor.cpu().numpy()  # (F, 3)

    # Write PLY
    num_verts = vertices.shape[0]
    num_faces = faces.shape[0]

    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {num_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"[Pathtracer] Generated PLY mesh: {ply_path} "
          f"({num_verts} verts, {num_faces} faces)")


class RayTracer:
    def __init__(self, smplx_model_path='../models/smplx_models',
                 gender='neutral') -> None:
        self.PIR_resolution = 128
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load SMPL-X body model (52 joints)
        self.body = smpl.get_smplx_layer(
            model_path=smplx_model_path,
            gender=gender,
        )

        # Generate PLY from SMPL-X for Mitsuba scene
        ply_path = os.path.join(
            os.path.dirname(smplx_model_path), 'smplx_tpose.ply'
        )
        _ensure_ply_mesh(self.body, ply_path)

        self.scene = mi.load_dict(
            get_default_scene(res=self.PIR_resolution, ply_path=ply_path)
        )
        self.params_scene = mi.traverse(self.scene)

    def gen_rays(self):
        sensor = self.scene.sensors()[0]
        film = sensor.film()
        sampler = sensor.sampler()
        film_size = film.crop_size()
        spp = 1
        total_sample_count = dr.prod(film_size) * spp
        if sampler.wavefront_size() != total_sample_count:
            sampler.seed(0, total_sample_count)

        pos = dr.arange(mi.UInt32, total_sample_count)
        pos //= spp
        scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
        pos = mi.Vector2f(
            mi.Float(pos % int(film_size[0])),
            mi.Float(pos // int(film_size[0])),
        )
        rays, weights = sensor.sample_ray_differential(
            time=0,
            sample1=sampler.next_1d(),
            sample2=pos * scale,
            sample3=0,
        )
        return rays

    def update_pose(self, pose_params, shape_params, translation=None):
        """
        Update the SMPL-X mesh in the Mitsuba scene.

        Args:
            pose_params: (D,) axis-angle pose vector
                         D=156 for full SMPL-X/SMPL-H (52 joints)
                         D=72 for SMPL-24 (hands will be flat)
            shape_params: (10,) body shape betas
            translation: (3,) or None, world-space translation
        """
        pose_params = torch.tensor(pose_params).view(1, -1)
        shape_params = torch.tensor(shape_params).view(1, -1)

        transform = None
        if translation is not None:
            transform = mi.Transform4f.translate(translation)
            transform = torch.tensor(transform.matrix).squeeze()

        vertices_mi = smpl.call_smplx_layer(
            pose_params, shape_params, self.body,
            need_face=False, transform=transform,
        )

        self.params_scene['smpl.vertex_positions'] = dr.ravel(vertices_mi)
        self.params_scene.update()

    def update_sensor(self, origin, target):
        transform = mi.Transform4f.look_at(
            origin=origin,
            target=target,
            up=(0, 1, 0),
        )
        self.params_scene['sensor.to_world'] = transform
        self.params_scene['tx.to_world'] = transform
        self.params_scene.update()

    def trace(self):
        ray = self.gen_rays()
        si = self.scene.ray_intersect(ray)
        intensity = mi.render(self.scene, spp=32)
        t = si.t
        t[t > 9999] = 0
        distance = np.array(t).reshape(self.PIR_resolution, self.PIR_resolution)
        intensity = np.array(intensity)[:, :, 0]
        velocity = np.zeros((self.PIR_resolution, self.PIR_resolution))

        PIR = np.stack([distance, intensity, velocity], axis=2)
        pointclouds = np.array(si.p)
        return PIR, pointclouds


def get_default_scene(res=512, ply_path='../models/smplx_tpose.ply'):
    """
    Construct the default Mitsuba scene dictionary.

    Args:
        res: PIR resolution
        ply_path: Path to the SMPL-X PLY mesh file
    """
    integrator = mi.load_dict({
        'type': 'direct',
    })

    sensor = mi.load_dict({
        'type': 'perspective',
        'to_world': T.look_at(
            origin=(0, 1, 3),
            target=(0, 1, 0),
            up=(0, 1, 0),
        ),
        'fov': 60,
        'film': {
            'type': 'hdrfilm',
            'width': res,
            'height': res,
            'rfilter': {'type': 'gaussian'},
            'sample_border': True,
            'pixel_format': 'luminance',
            'component_format': 'float32',
        },
        'sampler': {
            'type': 'independent',
            'sample_count': 1,
            'seed': 42,
        },
    })

    default_scene = {
        'type': 'scene',
        'integrator': integrator,
        'sensor': sensor,

        'while': {
            'type': 'diffuse',
            'reflectance': {'type': 'rgb', 'value': (0.8, 0.8, 0.8)},
        },
        'smpl': {
            'type': 'ply',
            'filename': ply_path,
            "mybsdf": {
                "type": "ref",
                "id": "while",
            },
        },

        'tx': {
            'type': 'spot',
            'cutoff_angle': 60,
            'to_world': T.look_at(
                origin=(0, 1, 3),
                target=(0, 1, 0),
                up=(0, 1, 0),
            ),
            'intensity': 1500.0,
        },
    }
    return default_scene


# Keep backward-compatible name
get_deafult_scene = get_default_scene


def trace(motion_filename):
    """
    Trace all frames of a motion sequence through the Mitsuba pathtracer.

    Supports both:
      - 156D pose (SMPL-X / SMPL-H, 52 joints including fingers)
      - 72D pose (SMPL-24, hands will be flat/identity)

    Args:
        motion_filename: Path to .npz file with keys:
            'pose': (N, D) axis-angle, D=156 or 72
            'shape': (10,) or (N, 10) body shape betas
            'root_translation': (N, 3) world translation per frame

    Returns:
        PIRs: list of torch tensors, each (H, W, 3) [distance, intensity, velocity]
        pointclouds: list of torch tensors, each (H*W, 3) world-space points
    """
    smpl_data = np.load(motion_filename, allow_pickle=True)
    root_translation = smpl_data['root_translation']

    # Determine gender
    gender = str(smpl_data.get('gender', 'neutral'))
    if isinstance(gender, np.ndarray):
        gender = str(gender)

    # Detect pose format
    pose_data = smpl_data['pose']
    num_pose_values = pose_data.shape[-1]
    if num_pose_values >= 156:
        joint_desc = "SMPL-X/SMPL-H 52 joints (156D)"
    elif num_pose_values >= 72:
        joint_desc = "SMPL-24 joints (72D) - hands will be flat"
    else:
        joint_desc = f"Unknown format ({num_pose_values}D)"
    print(f"[Trace] Pose format: {joint_desc}")

    raytracer = RayTracer(gender=gender)

    # Compute sensor placement from trajectory
    traj_center = root_translation.mean(axis=0)
    sensor_distance = 3.0
    sensor_origin = (
        float(traj_center[0]),
        float(traj_center[1] + 1.0),
        float(traj_center[2] + sensor_distance),
    )
    sensor_target = (
        float(traj_center[0]),
        float(traj_center[1] + 1.0),
        float(traj_center[2]),
    )

    raytracer.update_sensor(origin=sensor_origin, target=sensor_target)
    print(f"[Trace] sensor origin={sensor_origin}, target={sensor_target}")
    print(f"[Trace] trajectory center={traj_center}, "
          f"y range=[{root_translation[:, 1].min():.3f}, "
          f"{root_translation[:, 1].max():.3f}]")

    PIRs = []
    pointclouds = []
    total_motion_frames = len(root_translation)

    # Handle shape params: might be (10,) or (N, 10)
    shape_data = smpl_data['shape']
    if shape_data.ndim == 1:
        shape_per_frame = shape_data
    else:
        shape_per_frame = shape_data  # will index per frame

    for frame_idx in tqdm(range(total_motion_frames), desc="Rendering Body PIRs"):
        shape = shape_per_frame if shape_data.ndim == 1 else shape_per_frame[frame_idx]

        raytracer.update_pose(
            smpl_data['pose'][frame_idx],
            shape,
            np.array(root_translation[frame_idx]),
        )
        PIR, pc = raytracer.trace()
        PIRs.append(torch.from_numpy(PIR).cuda())
        pointclouds.append(torch.from_numpy(pc).cuda())

    return PIRs, pointclouds
