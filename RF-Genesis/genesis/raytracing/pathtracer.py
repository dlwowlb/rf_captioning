import drjit as dr
import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T
import numpy as np
from . import smpl
import torch
from tqdm import tqdm
mi.set_variant('cuda_ad_rgb')
torch.set_default_device('cuda')


class RayTracer:
    def __init__(self) -> None:
        self.PIR_resolution = 128
        self.scene = mi.load_dict(get_deafult_scene(res = self.PIR_resolution))
        self.params_scene = mi.traverse(self.scene)
        self.body = smpl.get_smpl_layer()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        pos = mi.Vector2f(mi.Float(pos  % int(film_size[0])),
                    mi.Float(pos // int(film_size[0])))
        rays, weights = sensor.sample_ray_differential(
            time=0,
            sample1=sampler.next_1d(),
            sample2=pos * scale,
            sample3=0
        )
        return rays
    
    def update_pose(self,pose_params, shape_params, translation= None):
        
        pose_params = torch.tensor(pose_params).view(1, -1)
        shape_params = torch.tensor(shape_params).view(1, -1)

        if translation is not None:
            transform = mi.Transform4f.translate(translation)
            transform = torch.tensor(transform.matrix).squeeze()


        vertices_mi=smpl.call_smpl_layer(pose_params,shape_params,self.body,need_face=False,transform=transform)
        
        self.params_scene['smpl.vertex_positions'] = dr.ravel(vertices_mi)
        self.params_scene.update()
    
    def update_sensor(self,origin, target):
        transform = mi.Transform4f.look_at(
                            origin=origin,
                            target=target,
                            up=(0, 1, 0)
                        )
        self.params_scene['sensor.to_world'] = transform
        self.params_scene['tx.to_world'] = transform
        self.params_scene.update()
    
    

    def trace(self):
        ray = self.gen_rays()
        si = self.scene.ray_intersect(ray)                   # ray intersection
        intensity = mi.render(self.scene,spp=32)
        t= si.t
        t[t>9999]=0
        distance = np.array(t).reshape(self.PIR_resolution,self.PIR_resolution)
        intensity = np.array(intensity)[:,:,0]
        velocity = np.zeros((self.PIR_resolution,self.PIR_resolution))  # the velocity is zero for this static frame, 
                                                                        # but will be calculated later by calculating the difference between two frames
        
        PIR = np.stack([distance,intensity,velocity],axis=2)
        pointclouds = np.array(si.p)        # We save the points here for faster calculation, it can be calculated from the PIR's distance + sensor's intrinsic metrix
        return PIR, pointclouds
    


def get_deafult_scene(res = 512):
    integrator = mi.load_dict({
        'type': 'direct',
        })

    sensor = mi.load_dict({
            'type': 'perspective',
            'to_world': T.look_at(
                            origin=(0, 1, 3),
                            target=(0, 1, 0),
                            up=(0, 1, 0)
                        ),
            'fov': 60,
            'film': {
                'type': 'hdrfilm',
                'width': res,
                'height': res,
                'rfilter': { 'type': 'gaussian' },
                'sample_border': True,
                'pixel_format': 'luminance',
                'component_format': 'float32',
            },
            'sampler':{
                'type': 'independent',
                'sample_count': 1,
                'seed':42
            },
        })


    default_scene ={
            'type': 'scene',
            'integrator': integrator,
            'sensor': sensor,
            
            'while':{
                'type':'diffuse',
                'reflectance': { 'type': 'rgb', 'value': (0.8, 0.8, 0.8) }, 
            },
            'smpl':{
                'type': 'ply',
                'filename': '../models/male.ply',
                "mybsdf": {
                    "type": "ref",
                    "id": "while"
                },
            },

            'tx':{
                'type': 'spot',
                'cutoff_angle': 40,
                'to_world': T.look_at(
                                origin=(0, 0, 3),
                                target=(0, 0, 0),
                                up=(0, 1, 0)
                            ),
                'intensity': 1000.0,
            }

        }
    return default_scene



def trace(motion_filename):
    smpl_data = np.load(motion_filename, allow_pickle=True)
    root_translation = smpl_data['root_translation']
    max_distance = np.max(root_translation[:,2])+2
    body_offset = np.array([0,1,3])
    sensor_origin = np.array([0,0,0])
    sensor_target = np.array([0,0,-5])

    # Check for SMPL-H hand data
    has_keypoints3d = 'keypoints3d' in smpl_data
    has_pose_smplh = 'pose_smplh' in smpl_data

    if has_keypoints3d:
        keypoints3d = smpl_data['keypoints3d']  # (N, 52, 3)
        num_hand_joints = keypoints3d.shape[1] - 22 if keypoints3d.shape[1] > 22 else 0
        print(f"[PathTracer] Found keypoints3d with {keypoints3d.shape[1]} joints ({num_hand_joints} hand joints)")
    elif has_pose_smplh:
        print("[PathTracer] Found pose_smplh, will compute hand keypoints via FK")
        from genesis.object_diffusion.object_diff import compute_hand_keypoints3d
        keypoints3d = compute_hand_keypoints3d(
            smpl_data['pose_smplh'],
            smpl_data['root_translation'],
        )
        num_hand_joints = 30
    else:
        keypoints3d = None
        num_hand_joints = 0

    raytracer = RayTracer()
    PIRs = []
    pointclouds = []
    total_motion_frames = len(root_translation)

    for frame_idx in tqdm(range(0, total_motion_frames), desc="Rendering Body PIRs"):
        translation = np.array(root_translation[frame_idx]) - body_offset
        raytracer.update_pose(smpl_data['pose'][frame_idx], smpl_data['shape'][0], translation)
        PIR, pc = raytracer.trace()

        # Append hand joint positions to point clouds if available
        if keypoints3d is not None and keypoints3d.shape[1] > 22:
            hand_joints = keypoints3d[frame_idx, 22:]  # (30, 3)
            # Apply same body_offset transform as mesh vertices
            hand_joints_shifted = hand_joints - body_offset
            # Convert to same format as pc: each row is (x, y, z)
            hand_pc = hand_joints_shifted.astype(np.float32)
            pc_combined = np.concatenate([np.array(pc), hand_pc], axis=0)
            pointclouds.append(torch.from_numpy(pc_combined).cuda())
        else:
            pointclouds.append(torch.from_numpy(np.array(pc)).cuda())

        PIRs.append(torch.from_numpy(PIR).cuda())

    return PIRs, pointclouds
