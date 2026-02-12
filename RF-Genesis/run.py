import argparse
from termcolor import colored
import time
from genesis.raytracing import pathtracer
from genesis.raytracing import signal_generator
from genesis.raytracing.wall import Wall, WallMaterial, MATERIAL_PRESETS

from genesis.environment_diffusion import environemnt_diff
from genesis.object_diffusion import object_diff
from genesis.visualization import visualize


import torch
import numpy as np
import os
torch.set_default_device('cuda')

def get_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='List the content of a folder')
    parser.add_argument('-o', '--obj-prompt', type=str, help='Specify the object prompt')
    parser.add_argument('-e','--env-prompt', type=str, help='Specify the environment prompt')
    parser.add_argument('-n', '--name', type=str, help='Specify the name (optional)')
    parser.add_argument('--no-visualize',  dest='skip_visualize', default= False,
                        help='Disable visualization step (default: enabled)')
    parser.add_argument('--no-environment',  dest='skip_environment', default= False,
                        help='Disable environment PIR generation (default: enabled)')

    # Through-wall simulation options
    parser.add_argument('--wall-material', type=str, default=None,
                        choices=list(MATERIAL_PRESETS.keys()),
                        help='Wall material type for through-wall simulation. '
                             'Available: ' + ', '.join(MATERIAL_PRESETS.keys()))
    parser.add_argument('--wall-thickness', type=float, default=0.1,
                        help='Wall thickness in meters (default: 0.1 = 10cm)')
    parser.add_argument('--wall-distance', type=float, default=1.5,
                        help='Distance from sensor to wall in meters (default: 1.5)')
    parser.add_argument('--wall-epsilon-r', type=float, default=None,
                        help='Custom wall relative permittivity (overrides --wall-material)')
    parser.add_argument('--wall-loss-tangent', type=float, default=None,
                        help='Custom wall loss tangent (overrides --wall-material)')

    args = parser.parse_args()

    return (args.obj_prompt, args.env_prompt, args.name,
            args.skip_visualize, args.skip_environment,
            args.wall_material, args.wall_thickness, args.wall_distance,
            args.wall_epsilon_r, args.wall_loss_tangent)



def run_pipeline(obj_prompt, env_prompt=None, name=None, skip_visualize=False,
                 skip_environment=False, hymotion_output=None,
                 walls=None, wall_spec=None):
    """Run the RF-Genesis pipeline.

    Args:
        obj_prompt: text prompt for motion generation
        env_prompt: text prompt for environment generation
        name: output directory name
        skip_visualize: whether to skip visualization
        skip_environment: whether to skip environment generation
        hymotion_output: pre-generated HY-Motion output dict with 'rot6d' and 'transl' keys.
                         When provided, uses HY-Motion instead of MDM for motion generation.
        walls: list of Wall objects for through-wall simulation, or None.
               Use this when calling run_pipeline() from code with pre-built Wall objects.
        wall_spec: tuple of (WallMaterial, thickness, distance) from CLI args.
                   The Wall object will be created once sensor position is known.
    """
    

    if name is None:
        name = f"output_{int(time.time())}"

    output_dir = os.path.join("output", name)
    os.makedirs(output_dir, exist_ok=True)


    if not os.path.exists(os.path.join(output_dir, 'obj_diff.npz')):
        print(colored('[RFGen] Step 1/4: Generating the human body motion: ', 'green'))
        object_diff.generate(obj_prompt, output_dir, hymotion_output=hymotion_output)
    else:
        print(colored('[RFGen] Step 1/4: Already done, existing body motion file, skiping this step.', 'green'))

    
    os.chdir("genesis/")
    print(colored('[RFGen] Step 2/4: Rendering the human body PIRs: ', 'green'))
    
    motion_path = os.path.join("../", output_dir, 'obj_diff.npz')
    body_pir, body_aux = pathtracer.trace(motion_path)
 
    # Read back sensor position computed by pathtracer.trace()
    smpl_data = np.load(motion_path, allow_pickle=True)
    root_translation = smpl_data['root_translation']
    traj_center = root_translation.mean(axis=0)
    sensor_origin = [
        float(traj_center[0]),
        float(traj_center[1] + 1.0),
        float(traj_center[2] + 3.0),
    ]
    sensor_target = [
        float(traj_center[0]),
        float(traj_center[1] + 1.0),
        float(traj_center[2]),
    ]

    os.chdir("..")

    # Build Wall objects once sensor position is known
    if wall_spec is not None and walls is None:
        material, thickness, distance = wall_spec
        # Place the wall between sensor and body, perpendicular to viewing direction.
        # Sensor is at sensor_origin, body center at sensor_target.
        # Wall is at sensor_origin + distance * (-z_direction) where z_direction
        # points from sensor toward target.
        direction = np.array(sensor_target) - np.array(sensor_origin)
        direction_norm = direction / np.linalg.norm(direction)
        wall_pos = np.array(sensor_origin) + distance * direction_norm
        # Wall normal points toward sensor (+z = back toward sensor)
        wall_normal = -direction_norm
        walls = [Wall(
            position=wall_pos,
            normal=wall_normal,
            thickness=thickness,
            material=material,
        )]

    if not skip_environment:
        print(colored('[RFGen] Step 3/4: Generating the environmental PIRs: ', 'green'))
        envir_diff = environemnt_diff.EnvironmentDiffusion(lora_path="Asixa/RFLoRA")
        env_pir = envir_diff.generate(env_prompt)
    else:
        print(colored('[RFGen] Step 3/4: Skipping environment generation as requested.', 'yellow'))
        env_pir = None


    print(colored('[RFGen] Step 4/4: Generating the radar signal.', 'green'))

    if walls:
        for w in walls:
            print(colored(f'[RFGen] Through-wall: {w.summary()}', 'cyan'))

    radar_frames = signal_generator.generate_signal_frames(
        body_pir, body_aux, env_pir, radar_config="models/TI1843_config.json",
        sensor_origin=sensor_origin, sensor_target=sensor_target,
        walls=walls,
    )

    print(colored('[RFGen] Saving the radar bin file. Shape {}'.format(radar_frames.shape), 'green'))
    np.save(os.path.join(output_dir, 'radar_frames.npy'), radar_frames)

    if not skip_visualize:
        print(colored('[RFGen] Rendering the visualization.', 'green'))
        torch.set_default_device('cpu')  # To avoid OOM
        visualize.save_video(
            "models/TI1843_config.json",
            os.path.join(output_dir, 'radar_frames.npy'),
            os.path.join(output_dir, 'obj_diff.npz'),
            os.path.join(output_dir, 'output.mp4'))
    else:
        print(colored('[RFGen] Skipping visualization step.', 'yellow'))


    print(colored('----------------------------------------', 'green')) 
    print(colored('[RFGen] Hooray! you are all set! ', 'green')) 
    print(colored('----------------------------------------', 'green')) 
    print(colored('        Please ignore the segmentation faults if there are any.', 'green'))


    return output_dir
 
 
def build_walls_from_args(wall_material, wall_thickness, wall_distance,
                          wall_epsilon_r, wall_loss_tangent):
    """Create Wall objects from CLI arguments.

    The wall is placed between the sensor and the body, perpendicular to
    the sensor's viewing direction (z-axis). The wall normal points toward
    the sensor (+z direction).

    The wall position is computed later in run_pipeline once we know
    the sensor position. Here we return a factory function.
    """
    if wall_material is None and wall_epsilon_r is None:
        return None

    # Determine material properties
    if wall_epsilon_r is not None and wall_loss_tangent is not None:
        material = WallMaterial(
            name=f'Custom(er={wall_epsilon_r}, tand={wall_loss_tangent})',
            epsilon_r=wall_epsilon_r,
            loss_tangent=wall_loss_tangent,
        )
    elif wall_material is not None:
        material = MATERIAL_PRESETS[wall_material]
    else:
        raise ValueError("Specify --wall-material or both --wall-epsilon-r and --wall-loss-tangent")

    return material, wall_thickness, wall_distance


def main():
    (obj_prompt, env_prompt, name, skip_visualize, skip_environment,
     wall_material, wall_thickness, wall_distance,
     wall_epsilon_r, wall_loss_tangent) = get_args()

    # Build wall specification (actual Wall object created in run_pipeline
    # after sensor position is known)
    wall_spec = build_walls_from_args(
        wall_material, wall_thickness, wall_distance,
        wall_epsilon_r, wall_loss_tangent)

    # We need to defer Wall creation until sensor position is known.
    # Pass wall_spec to run_pipeline as a simple structure.
    run_pipeline(obj_prompt, env_prompt, name, skip_visualize, skip_environment,
                 wall_spec=wall_spec)

    exit(0)
if __name__ == '__main__':
    main()


    
