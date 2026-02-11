import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os

# ==========================================
# 1. 안테나 배치 확인 (축 교정 버전)
# ==========================================
def check_radar_geometry(config_path):
    if not os.path.exists(config_path):
        print(f"[Error] Config file not found: {config_path}")
        tx_loc = [[0,0,0], [2e-3, 0, 0], [4e-3, 0, 0]]
        rx_loc = [[0,0,0], [0, 2e-3, 0], [0, 4e-3, 0], [0, 6e-3, 0]]
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        c = 3e8
        fc = config.get('fc', 77e9)
        wavelength = c / fc
        spacing = wavelength / 2
        
        tx_loc = np.array(config['tx_loc']) * spacing
        rx_loc = np.array(config['rx_loc']) * spacing

    tx_loc = np.array(tx_loc) * 1000 # mm 단위
    rx_loc = np.array(rx_loc) * 1000

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # ★ [핵심 수정] Data Y(높이) -> Plot Z(수직)로 매핑
    # Plot 순서: (Data X, Data Z, Data Y)
    
    # Tx (송신)
    ax.scatter(tx_loc[:,0], tx_loc[:,2], tx_loc[:,1], c='r', marker='^', s=150, label='Tx')
    # Rx (수신)
    ax.scatter(rx_loc[:,0], rx_loc[:,2], rx_loc[:,1], c='b', marker='o', s=150, label='Rx')

    # Virtual Array
    virtual_locs = []
    for t in tx_loc:
        for r in rx_loc:
            virtual_locs.append(t + r)
    virtual_locs = np.array(virtual_locs)
    
    ax.scatter(virtual_locs[:,0], virtual_locs[:,2], virtual_locs[:,1], 
               c='g', alpha=0.3, s=50, label='Virtual Array')

    # 축 라벨 수정 (Matplotlib 기준)
    ax.set_xlabel('Radar X (Horizontal) [mm]')
    ax.set_ylabel('Radar Z (Depth) [mm]')
    ax.set_zlabel('Radar Y (Vertical/Height) [mm]') # Z축에 Y라벨 붙임
    
    ax.set_title("Antenna Geometry (Upright View)")
    ax.legend()
    
    # 비율 맞추기
    all_vals = np.concatenate([tx_loc, rx_loc, virtual_locs])
    max_range = np.ptp(all_vals) / 2.0
    mid_x, mid_y, mid_z = np.mean(all_vals, axis=0)
    
    # Plot 중심 이동 (Data X, Data Z, Data Y 순서)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_z - max_range, mid_z + max_range)
    ax.set_zlim(mid_y - max_range, mid_y + max_range)
    
    plt.show()

# ==========================================
# 2. 전체 씬 확인 (축 교정 버전)
# ==========================================
def check_motion_trajectory_corrected(npz_path):
    if not os.path.exists(npz_path):
        print(f"[Error] Motion file not found: {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)
    if 'root_translation' in data:
        traj = data['root_translation']
    elif 'transl' in data:
        traj = data['transl']
    else:
        print("Cannot find translation data.")
        return

    traj_center = traj.mean(axis=0)
    
    # [설정] 사용자님의 코드와 동일하게 설정 (골반 높이 정면)
    sensor_origin = np.array([traj_center[0], traj_center[1], traj_center[2] + 3.0])

    print(f"[*] Person Center: {traj_center}")
    print(f"[*] Radar Position: {sensor_origin}")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ★ [핵심 수정] Data (x, y, z) -> Plot (x, z, y) 매핑
    # Data Y(높이)를 Plot Z(수직축)에 넣어야 서 있는 것처럼 보임

    # 사람 경로
    ax.plot(traj[:,0], traj[:,2], traj[:,1], c='blue', linewidth=2, label='Path')
    ax.scatter(traj[0,0], traj[0,2], traj[0,1], c='green', marker='^', s=100, label='Start')
    ax.scatter(traj[-1,0], traj[-1,2], traj[-1,1], c='red', marker='x', s=100, label='End')

    # 레이더 위치
    ax.scatter(sensor_origin[0], sensor_origin[2], sensor_origin[1], 
               c='black', marker='s', s=300, label='Radar')
    
    # 시야선 (Line of Sight)
    ax.plot([sensor_origin[0], traj_center[0]], 
            [sensor_origin[2], traj_center[2]], 
            [sensor_origin[1], traj_center[1]], 'k--', alpha=0.3)

    # 바닥면 (Data Y = 0 인 지점) -> Plot Z = 0
    # 바닥을 그리기 위해 Plot X, Plot Y 범위를 잡음
    grid_min = min(traj[:,0].min(), sensor_origin[0]) - 1
    grid_max = max(traj[:,0].max(), sensor_origin[0]) + 1
    grid_depth_min = min(traj[:,2].min(), sensor_origin[2]) - 1
    grid_depth_max = max(traj[:,2].max(), sensor_origin[2]) + 1
    
    xx, yy = np.meshgrid(np.linspace(grid_min, grid_max, 10),
                         np.linspace(grid_depth_min, grid_depth_max, 10))
    zz = np.zeros_like(xx) # Data Y=0 (높이 0)
    
    ax.plot_surface(xx, yy, zz, color='gray', alpha=0.2)

    # 축 라벨
    ax.set_xlabel('World X (Left/Right) [m]')
    ax.set_ylabel('World Z (Depth) [m]')
    ax.set_zlabel('World Y (Height) [m]') # Z축에 Y라벨
    
    ax.set_title("Simulation Scene (Corrected Upright View)")
    
    # 뷰 조정 (쿼터뷰)
    ax.view_init(elev=20, azim=-45)
    
    # 비율 유지 (Zoom Out 포함)
    all_points = np.vstack([traj, sensor_origin.reshape(1,3)])
    # 축 순서 변경 (x, z, y)
    all_points_plot = all_points[:, [0, 2, 1]] 
    
    mid = np.mean(all_points_plot, axis=0)
    max_range = np.ptp(all_points_plot, axis=0).max() / 2.0
    
    ZOOM = 1.5
    radius = max_range * ZOOM
    
    ax.set_xlim(mid[0] - radius, mid[0] + radius)
    ax.set_ylim(mid[1] - radius, mid[1] + radius)
    ax.set_zlim(mid[2] - radius, mid[2] + radius)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 경로 설정 (본인 환경에 맞게 수정)
    config_file = "RF-Genesis/models/TI1843_config.json"
    motion_file = "output/test/run_20260211_202148/obj_diff.npz" 

    #print(">>> Figure 1: Antenna Layout (Close window to see next)")
    #check_radar_geometry(config_file)
    
    print(">>> Figure 2: Scene Layout")
    check_motion_trajectory_corrected(motion_file)
