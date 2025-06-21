#!/usr/bin/env python3
"""
3D Ellipsoid Visualization from Keypoints
Reads 3D keypoints, converts to bounding boxes, then visualizes as ellipsoids
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as pltcolors
from tqdm import tqdm

# Add the current directory to path to import crossview modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crossview_dataset.data_utils import Pose3DLoader
from crossview_dataset.calib.calibration import Calibration


def normalize_keypoints_simple(keypoints, calibration):
    """
    Simple normalization like VisPy visualizer - chỉ trừ world center
    """
    if len(keypoints) == 0:
        return keypoints
        
    keypoints = np.array(keypoints)
    
    # Lấy world center như trong VisPy
    world_ltrb = calibration.world_ltrb
    ori_wcx = np.mean(world_ltrb[0::2])  # world center X 
    ori_wcy = np.mean(world_ltrb[1::2])  # world center Y
    
    # Chỉ trừ center cho X,Y - giữ nguyên Z như VisPy
    normalized = keypoints.copy()
    normalized[:, 0] -= ori_wcx  # Center X around origin
    normalized[:, 1] -= ori_wcy  # Center Y around origin  
    # Z giữ nguyên - không thay đổi
    
    return normalized
    
    return normalized


def keypoints_to_bbox(keypoints, min_size=0.3):
    """
    Convert 3D keypoints to 3D bounding box
    
    Args:
        keypoints: numpy array of shape (N, 3) or (N, 4) with x, y, z, (score)
        min_size: minimum size for each dimension of bbox
        
    Returns:
        bbox: [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    if len(keypoints) == 0:
        return None
        
    keypoints = np.array(keypoints)
    
    # Filter out invalid keypoints (0,0,0) first
    xyz = keypoints[:, :3]
    valid_mask = ~np.all(xyz == 0, axis=1)  # Remove [0,0,0] points
    
    if not np.any(valid_mask):
        return None
    
    valid_keypoints = keypoints[valid_mask]
    
    # Filter valid keypoints if scores are available
    if valid_keypoints.shape[1] == 4:
        valid_keypoints = valid_keypoints[valid_keypoints[:, 3] > 0.1]  # confidence threshold
    
    if len(valid_keypoints) == 0:
        return None
        
    xyz = valid_keypoints[:, :3]
    x_min, y_min, z_min = xyz.min(axis=0)
    x_max, y_max, z_max = xyz.max(axis=0)
    
    # Ensure minimum size
    center_x, center_y, center_z = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    
    width = max(x_max - x_min, min_size)
    height = max(y_max - y_min, min_size)
    depth = max(z_max - z_min, min_size)
    
    # Limit Z dimension to reasonable human height (max 2 meters)
    depth = min(depth, 2.0)
    
    x_min = center_x - width / 2
    x_max = center_x + width / 2
    y_min = center_y - height / 2
    y_max = center_y + height / 2
    z_min = center_z - depth / 2
    z_max = center_z + depth / 2
    
    return [x_min, y_min, z_min, x_max, y_max, z_max]


def bbox_to_ellipsoid(bbox):
    """
    Convert 3D bounding box to ellipsoid parameters
    
    Args:
        bbox: [x_min, y_min, z_min, x_max, y_max, z_max]
        
    Returns:
        center: [cx, cy, cz]
        radii: [rx, ry, rz]
    """
    x_min, y_min, z_min, x_max, y_max, z_max = bbox
    
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    cz = (z_min + z_max) / 2
    
    rx = (x_max - x_min) / 2
    ry = (y_max - y_min) / 2
    rz = (z_max - z_min) / 2
    
    return [cx, cy, cz], [rx, ry, rz]


def get_skeleton_connections():
    """Define skeleton connections for human pose (COCO-style)"""
    # COCO 17-point skeleton connections
    connections = [
        (0, 1), (0, 2),    # nose to eyes
        (1, 3), (2, 4),    # eyes to ears
        (5, 6),            # shoulders
        (5, 7), (7, 9),    # left arm
        (6, 8), (8, 10),   # right arm
        (5, 11), (6, 12),  # torso
        (11, 12),          # hips
        (11, 13), (13, 15), # left leg
        (12, 14), (14, 16)  # right leg
    ]
    return connections


def get_color(idx):
    """Generate consistent colors for tracking IDs"""
    # Use a more diverse color palette
    colors = [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 1.0),  # Magenta
        (0.0, 1.0, 1.0),  # Cyan
        (1.0, 0.5, 0.0),  # Orange
        (0.5, 0.0, 1.0),  # Purple
        (0.0, 0.5, 0.0),  # Dark Green
        (0.5, 0.5, 0.0),  # Olive
    ]
    return colors[idx % len(colors)]


def plot_ellipsoid_3d(ax, center, radii, color='b', alpha=0.3):
    """
    Plot 3D ellipsoid on matplotlib axis
    
    Args:
        ax: matplotlib 3D axis
        center: [cx, cy, cz]
        radii: [rx, ry, rz]
        color: color for the ellipsoid
        alpha: transparency    """
    cx, cy, cz = center
    rx, ry, rz = radii
      # Make sure radii are not too small or too large  
    rx = max(min(rx, 5.0), 1.0)  # Increase minimum size to 1m, max 5m
    ry = max(min(ry, 5.0), 1.0)  # Increase minimum size to 1m, max 5m  
    rz = max(min(rz, 3.0), 1.5)  # Increase minimum height to 1.5m, max 3m
    
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    
    x = rx * np.outer(np.cos(u), np.sin(v)) + cx
    y = ry * np.outer(np.sin(u), np.sin(v)) + cy
    z = rz * np.outer(np.ones_like(u), np.cos(v)) + cz
    
    # Use surface plot instead of wireframe for better visibility
    ax.plot_surface(x, y, z, color=color, alpha=0.8, linewidth=1.0, edgecolor='black')


def visualize_ellipsoids_from_keypoints(calibration, pose_data, output_path="./results/ellipsoid_keypoints.mp4", 
                                      total_frames=None):
    """
    Create 3D ellipsoid visualization video from keypoints data
    
    Args:
        calibration: Calibration object with world_ltrb
        pose_data: Dictionary with frame-wise pose data (from Pose3DLoader)
        output_path: Output video path
        total_frames: Total number of frames to process    """
    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if there is one
        os.makedirs(output_dir, exist_ok=True)    # Video settings
    size = (1280, 720)
    fps = 5
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    print(f"Creating ellipsoid visualization video: {output_path}")
    
    # Get world bounds from calibration
    world_ltrb = calibration.world_ltrb
    x_min, y_min, x_max, y_max = world_ltrb    # Setup matplotlib figure
    fig = plt.figure(figsize=(size[0]/100, size[1]/100))
    ax = fig.add_subplot(111, projection='3d')
    
    # Store trajectory for each person
    person_trajectories = {}
    
    # Process each frame
    max_frames = total_frames or len(pose_data)
    for frame_idx in tqdm(range(max_frames), desc="Processing frames"):
        ax.clear()        # Set up 3D plot
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        
        # Set camera view angle for better visualization
        ax.view_init(elev=20, azim=45)  # Elevation 20°, Azimuth 45°        # Set world bounds - sử dụng centered world bounds như VisPy
        world_ltrb = calibration.world_ltrb
        ori_wcx = np.mean(world_ltrb[0::2]) 
        ori_wcy = np.mean(world_ltrb[1::2])
        
        # Centered world bounds
        x_min = world_ltrb[0] - ori_wcx
        y_min = world_ltrb[1] - ori_wcy  
        x_max = world_ltrb[2] - ori_wcx
        y_max = world_ltrb[3] - ori_wcy
        
        offset = 100  # Larger offset for better view
        ax.set_xlim3d(x_min - offset, x_max + offset)
        ax.set_ylim3d(y_min - offset, y_max + offset)
        ax.set_zlim3d(0, 200)  # Allow higher Z values like the original data        # Draw centered world boundary on ground
        ax.plot([x_min, x_max, x_max, x_min, x_min], 
                [y_min, y_min, y_max, y_max, y_min], 
                [0, 0, 0, 0, 0], 'k-', linewidth=2, alpha=0.5)
        
        # Debug info
        ellipsoid_count = 0
        
        # Get pose data for this frame
        if frame_idx < len(pose_data):
            frame_data = pose_data[frame_idx]            # Process each person in the frame
            for person_idx, person in enumerate(frame_data.get('poses', [])):
                person_id = person.get('id', -1)
                keypoints_3d = person.get('points_3d', [])
                
                if frame_idx < 3:  # Debug first few frames
                    print(f"Frame {frame_idx}, Person {person_id}:")
                    print(f"  Raw keypoints shape: {np.array(keypoints_3d).shape}")
                    if len(keypoints_3d) > 0:
                        kp_array = np.array(keypoints_3d)
                        print(f"  Original X: [{kp_array[:, 0].min():.1f}, {kp_array[:, 0].max():.1f}]")
                        print(f"  Original Y: [{kp_array[:, 1].min():.1f}, {kp_array[:, 1].max():.1f}]")
                        print(f"  Original Z: [{kp_array[:, 2].min():.1f}, {kp_array[:, 2].max():.1f}]")
                        
                        # Count valid keypoints
                        valid_count = np.sum(~np.all(kp_array == 0, axis=1))
                        print(f"  Valid keypoints: {valid_count}/{len(kp_array)}")                # Process keypoints - use simple normalization
                keypoints_3d_raw = np.array(keypoints_3d)
                
                # Use simple normalization instead of world coordinates
                keypoints_3d_normalized = normalize_keypoints_simple(keypoints_3d_raw, calibration)
                
                if frame_idx < 3:  # Debug normalized coordinates
                    print(f"  After normalization:")
                    if len(keypoints_3d_normalized) > 0:
                        print(f"    X range: [{keypoints_3d_normalized[:, 0].min():.1f}, {keypoints_3d_normalized[:, 0].max():.1f}]")
                        print(f"    Y range: [{keypoints_3d_normalized[:, 1].min():.1f}, {keypoints_3d_normalized[:, 1].max():.1f}]")
                        print(f"    Z range: [{keypoints_3d_normalized[:, 2].min():.1f}, {keypoints_3d_normalized[:, 2].max():.1f}]")
                
                # Filter out invalid keypoints (0,0,0)
                valid_mask = ~np.all(keypoints_3d_raw == 0, axis=1)
                if not np.any(valid_mask):
                    continue
                    
                valid_keypoints_normalized = keypoints_3d_normalized[valid_mask]
                  # Get unique color for this person using person_idx for consistency
                color_rgb = get_color(person_idx)
                
                # Plot keypoints as scatter points with better visibility
                ax.scatter(valid_keypoints_normalized[:, 0], valid_keypoints_normalized[:, 1], valid_keypoints_normalized[:, 2], 
                          c=[color_rgb], s=100, alpha=0.9, edgecolors='black', linewidths=1.5)
                
                # Draw skeleton connections with better validation
                connections = get_skeleton_connections()
                for connection in connections:
                    start_idx, end_idx = connection
                    # Check if both keypoints exist and are valid
                    if (start_idx < len(keypoints_3d_raw) and end_idx < len(keypoints_3d_raw) and
                        not np.all(keypoints_3d_raw[start_idx] == 0) and 
                        not np.all(keypoints_3d_raw[end_idx] == 0)):
                        
                        start_point = keypoints_3d_normalized[start_idx]
                        end_point = keypoints_3d_normalized[end_idx]
                        
                        # Draw line between keypoints
                        ax.plot([start_point[0], end_point[0]], 
                               [start_point[1], end_point[1]], 
                               [start_point[2], end_point[2]], 
                               color=color_rgb, linewidth=3, alpha=0.8)
                  # Calculate center for label
                center_x = np.mean(valid_keypoints_normalized[:, 0])
                center_y = np.mean(valid_keypoints_normalized[:, 1]) 
                center_z = np.max(valid_keypoints_normalized[:, 2]) + 0.3
                
                # Store trajectory
                if person_id not in person_trajectories:
                    person_trajectories[person_id] = []
                person_trajectories[person_id].append([center_x, center_y, center_z])
                
                # Draw trajectory (last 10 positions)
                if len(person_trajectories[person_id]) > 1:
                    trajectory = np.array(person_trajectories[person_id][-10:])  # Last 10 positions
                    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                           color=color_rgb, linewidth=2, alpha=0.6, linestyle='--')
                
                # Add person ID label with person's color
                ax.text(center_x, center_y, center_z, 
                       f'ID:{person_id}', fontsize=14, color='white', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor=color_rgb, alpha=0.9))
                
                ellipsoid_count += 1
        
        
        # Add title
        ax.set_title(f'3D Human Tracking - Frame {frame_idx}', fontsize=14)
        
        # Convert matplotlib figure to image
        plt.tight_layout()
        fig.canvas.draw()
        
        # Convert to numpy arrayprint
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Resize to target size
        img = cv2.resize(img, size)
        
        # Write frame to video
        out.write(img)
    
    # Cleanup
    plt.close(fig)
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="3D Ellipsoid Visualization from Keypoints")
    parser.add_argument(
        "--calibration", type=str, required=True,
        help="Camera calibration JSON file"
    )
    parser.add_argument(
        "--pose-file", type=str, required=True,
        help="3D pose keypoints file (annotation_3d.json or result_3d.json)"
    )
    parser.add_argument(
        "--output-video", type=str, default="./results/ellipsoid_keypoints.mp4",
        help="Output path for 3D visualization video"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum number of frames to process"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        # Load calibration
        calibration = Calibration.from_json(args.calibration)
        print(f"Loaded calibration from {args.calibration}")
        print(f"World bounds: {calibration.world_ltrb}")
        
        # Load pose data
        pose_loader = Pose3DLoader(args.pose_file)
        print(f"Loaded 3D poses from {args.pose_file}")
        print(f"Total frames: {len(pose_loader)}")
        
        # Convert pose data to list
        pose_data = [pose_loader[i] for i in range(len(pose_loader))]
          # Debug: Print first frame data
        if len(pose_data) > 0:
            print(f"\nFirst frame debug:")
            print(f"Keys: {pose_data[0].keys()}")
            if 'poses' in pose_data[0]:
                print(f"Number of poses: {len(pose_data[0]['poses'])}")
                if len(pose_data[0]['poses']) > 0:
                    first_pose = pose_data[0]['poses'][0]
                    print(f"First pose keys: {first_pose.keys()}")
                    if 'points_3d' in first_pose:
                        keypoints = np.array(first_pose['points_3d'])
                        print(f"Raw keypoints range:")
                        print(f"  X: [{keypoints[:, 0].min():.2f}, {keypoints[:, 0].max():.2f}]")
                        print(f"  Y: [{keypoints[:, 1].min():.2f}, {keypoints[:, 1].max():.2f}]")
                        print(f"  Z: [{keypoints[:, 2].min():.2f}, {keypoints[:, 2].max():.2f}]")
                        
                        # Check if Z increases over time (which would be wrong)
                        if len(pose_data) > 10:
                            z_values = []
                            for i in range(min(10, len(pose_data))):
                                if 'poses' in pose_data[i] and len(pose_data[i]['poses']) > 0:
                                    kpts = np.array(pose_data[i]['poses'][0]['points_3d'])
                                    valid_z = kpts[kpts[:, 2] > 0, 2]
                                    if len(valid_z) > 0:
                                        z_values.append(np.mean(valid_z))
                            if len(z_values) > 5:
                                z_trend = np.polyfit(range(len(z_values)), z_values, 1)[0]
                                print(f"  Z trend over first 10 frames: {z_trend:.3f} (should be ~0, not increasing)")
                                if z_trend > 5:
                                    print("  WARNING: Z is increasing rapidly - coordinate system may be flipped!")
   
        # Generate visualization
        visualize_ellipsoids_from_keypoints(
            calibration=calibration,
            pose_data=pose_data,
            output_path=args.output_video,
            total_frames=args.max_frames
        )
        
        print(f"3D ellipsoid visualization completed successfully!")
        print(f"Output saved to: {args.output_video}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
