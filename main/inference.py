import os
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
# import mediapipe as mp
from tqdm import tqdm
from pathlib import Path
from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.inference_utils import non_max_suppression
from utils.distribute_utils import get_device

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d, matrix_to_axis_angle

os.environ['GLOG_minloglevel'] = '3'

# Initialize MediaPipe pose detection
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_pose = mp.solutions.pose

# def get_pelvis_center(landmarks):
#     """
#     Calculate pelvis center from MediaPipe landmarks.
#     In MediaPipe, the hip points are landmarks 23 (LEFT_HIP) and 24 (RIGHT_HIP).
#     The pelvis center is the midpoint between these two hip points.
    
#     Args:
#         landmarks: MediaPipe pose landmarks
        
#     Returns:
#         Tuple of (x, y, z) coordinates of the pelvis center in normalized coordinates
#     """
#     if not landmarks:
#         return None
        
#     # Get left and right hip landmarks
#     left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
#     right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    
#     # Calculate the midpoint (pelvis center)
#     pelvis_x = (left_hip.x + right_hip.x) / 2
#     pelvis_y = (left_hip.y + right_hip.y) / 2
#     pelvis_z = (left_hip.z + right_hip.z) / 2
    
#     return (pelvis_x, pelvis_y, pelvis_z)

# def get_lowest_keypoint_y(landmarks, img_height, y_min):
#     """
#     Find the maximum y-coordinate (lowest point) among all detected landmarks.

#     Args:
#         landmarks: MediaPipe pose landmarks object (results.pose_landmarks).
#         img_height: Height of the cropped image where landmarks were detected.
#         y_min: The y-offset of the crop in the original image.

#     Returns:
#         Maximum y-coordinate in original image space, or None if no landmarks found.
#     """
#     if not landmarks or not landmarks.landmark:
#         return None

#     max_y = -1
#     valid_landmark_found = False

#     # Iterate through all available landmarks
#     for landmark in landmarks.landmark:
#         # Basic check if landmark data is usable (y coordinate exists)
#         if hasattr(landmark, 'y'):
#             y_coord_cropped = landmark.y * img_height
#             y_coord_global = int(y_coord_cropped + y_min)
#             if y_coord_global > max_y:
#                 max_y = y_coord_global
#             valid_landmark_found = True

#     return max_y if valid_landmark_found else None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--file_name', type=str, default='test')
    parser.add_argument('--ckpt_name', type=str, default='model_dump')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--multi_person', action='store_true', default=False)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    return args

def interpolate_missing_motion(motion_list, missing_frames):
    """
    Interpolate missing frames in the motion_list based on surrounding frames.

    Args:
        motion_list: List of numpy arrays containing motion data for detected frames (can contain None).
        missing_frames: List of frame indices (relative to motion_list) that are missing.

    Returns:
        List: motion_list with interpolated frames filled in.
    """
    if not missing_frames:
        return motion_list # No interpolation needed

    num_frames_total = len(motion_list)
    # Create a copy to modify
    interpolated_motion_list = list(motion_list) 

    # Find the first and last valid frames for edge case handling
    first_valid_idx = -1
    for i in range(num_frames_total):
        if interpolated_motion_list[i] is not None:
            first_valid_idx = i
            break
            
    last_valid_idx = -1
    for i in range(num_frames_total - 1, -1, -1):
        if interpolated_motion_list[i] is not None:
            last_valid_idx = i
            break

    # Handle missing frames at the beginning
    if first_valid_idx > 0 and interpolated_motion_list[0] is None:
        fill_motion = interpolated_motion_list[first_valid_idx].copy()
        for i in range(first_valid_idx):
            if interpolated_motion_list[i] is None:
                interpolated_motion_list[i] = fill_motion.copy()

    # Handle missing frames at the end
    if last_valid_idx != -1 and last_valid_idx < num_frames_total - 1 and interpolated_motion_list[-1] is None:
        fill_motion = interpolated_motion_list[last_valid_idx].copy()
        for i in range(last_valid_idx + 1, num_frames_total):
             if interpolated_motion_list[i] is None:
                interpolated_motion_list[i] = fill_motion.copy()

    # Group consecutive missing frames
    missing_groups = []
    if not missing_frames: # Re-check after edge handling
        return interpolated_motion_list

    current_group = [missing_frames[0]]
    for i in range(1, len(missing_frames)):
        # Only consider frames between the first and last valid ones for interpolation
        if first_valid_idx <= missing_frames[i] <= last_valid_idx:
            if missing_frames[i] == missing_frames[i-1] + 1:
                current_group.append(missing_frames[i])
            else:
                if current_group: # Ensure group is not empty before adding
                     missing_groups.append(current_group)
                current_group = [missing_frames[i]]
        elif current_group: # Add the last group if we moved past the valid range
             missing_groups.append(current_group)
             current_group = [] # Reset group

    if current_group: # Add the final group if it exists
        missing_groups.append(current_group)

    # Interpolate each group of missing frames (within the valid range)
    for group in missing_groups:
        start_group_idx = group[0]
        end_group_idx = group[-1]

        # Find the nearest available frame *before* the group
        prev_available_idx = start_group_idx - 1
        while prev_available_idx >= 0 and interpolated_motion_list[prev_available_idx] is None:
            prev_available_idx -= 1

        # Find the nearest available frame *after* the group
        next_available_idx = end_group_idx + 1
        while next_available_idx < num_frames_total and interpolated_motion_list[next_available_idx] is None:
            next_available_idx += 1

        # Ensure we found valid frames for interpolation
        if prev_available_idx < 0 or next_available_idx >= num_frames_total:
            # This should ideally not happen after handling edge cases, but as a fallback:
            print(f"Warning: Could not find valid surrounding frames for group {group}. Skipping interpolation for this group.")
            continue 
            
        start_motion = interpolated_motion_list[prev_available_idx]
        end_motion = interpolated_motion_list[next_available_idx]
        
        # Linear interpolation using the indices of the found frames
        span = next_available_idx - prev_available_idx
        if span <= 0: 
             print(f"Warning: Invalid span {span} for interpolation group {group}. Skipping.")
             continue

        for frame_idx in group:
            # Ensure frame_idx is within the bounds defined by prev/next available
            if not (prev_available_idx < frame_idx < next_available_idx):
                 print(f"Warning: Frame index {frame_idx} out of bounds for interpolation span ({prev_available_idx}, {next_available_idx}). Skipping.")
                 continue

            alpha = (frame_idx - prev_available_idx) / float(span) # Use float division
            interpolated_motion = (1.0 - alpha) * start_motion + alpha * end_motion
            interpolated_motion_list[frame_idx] = interpolated_motion

    # Final check for any remaining None values (should ideally be none after edge handling)
    if any(m is None for m in interpolated_motion_list):
        print("Warning: Some None values remain after interpolation. Filling with nearest neighbor.")
        # Simple forward fill then backward fill
        last_valid = None
        for i in range(num_frames_total):
            if interpolated_motion_list[i] is not None:
                last_valid = interpolated_motion_list[i]
            elif last_valid is not None:
                interpolated_motion_list[i] = last_valid.copy()
        # Backward fill for any remaining Nones at the beginning
        first_valid = None
        for i in range(num_frames_total - 1, -1, -1):
             if interpolated_motion_list[i] is not None:
                 first_valid = interpolated_motion_list[i]
             elif first_valid is not None:
                 interpolated_motion_list[i] = first_valid.copy()


    return interpolated_motion_list

# def draw_mediapipe_pose(image, bbox, output_path):
#     """
#     Use MediaPipe to detect pose keypoints in the bounding box and save the visualized image.
#     Also returns the pelvis center coordinates and the lowest keypoint y coordinate if detected.

#     Args:
#         image: Input image
#         bbox: Bounding box coordinates [x, y, width, height]
#         output_path: Path to save the output image

#     Returns:
#         Tuple of:
#         - Annotated image with pose keypoints
#         - Pelvis center coordinates in original image space (x, y, z) or None if not detected
#         - Lowest keypoint y-coordinate in original image space or None if not detected
#     """
#     # Ensure the image is in uint8 format (fix for TypeError)
#     if image.dtype != np.uint8:
#         # If image is float32 with values in [0,1] range
#         if image.max() <= 1.0:
#             image = (image * 255).astype(np.uint8)
#         # If image is already in the right range but wrong type
#         else:
#             image = image.astype(np.uint8)
    
#     # Crop to bounding box with some padding
#     x, y, w, h = [int(v) for v in bbox]
    
#     # Add padding to ensure the entire person is in view
#     padding = 0 # Restore padding
#     x_min = max(0, x - padding)
#     y_min = max(0, y - padding)
#     x_max = min(image.shape[1], x + w + padding)
#     y_max = min(image.shape[0], y + h + padding)
    
#     # Crop the image to the bounding box
#     cropped_image = image[y_min:y_max, x_min:x_max]

#     # Skip if the cropped image is empty
#     if cropped_image.size == 0:
#         return image, None, None # Return None for max_foot_y as well

#     pelvis_data = None
#     lowest_y_global = None # Renamed from max_foot_y_global

#     # Calculate bbox bottom y in original image space
#     bbox_bottom_y_global = y + h
        
#     # Process the image with MediaPipe
#     with mp_pose.Pose(
#         static_image_mode=True,  # Set to True for images
#         model_complexity=2,  # Use the most accurate model
#         enable_segmentation=False,
#         min_detection_confidence=0.5) as pose:
        
#         # Convert the image to RGB for MediaPipe
#         results = pose.process(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        
#         # Draw pose landmarks on the cropped image
#         annotated_cropped = cropped_image.copy()
#         if results.pose_landmarks:
#             mp_drawing.draw_landmarks(
#                 annotated_cropped,
#                 results.pose_landmarks,
#                 mp_pose.POSE_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
#             # Get pelvis center from landmarks
#             pelvis_center_norm = get_pelvis_center(results.pose_landmarks)
#             if pelvis_center_norm:
#                 # Convert normalized coordinates to pixel coordinates in the cropped image
#                 pelvis_x = int(pelvis_center_norm[0] * cropped_image.shape[1])
#                 pelvis_y = int(pelvis_center_norm[1] * cropped_image.shape[0])
#                 pelvis_z = pelvis_center_norm[2]  # Keep z in normalized format
                
#                 # Convert coordinates back to original image space
#                 pelvis_x_global = pelvis_x + x_min
#                 pelvis_y_global = pelvis_y + y_min # Return actual pelvis y
                
#                 # Store full pelvis data including z
#                 pelvis_data = (pelvis_x_global, pelvis_y_global, pelvis_z)
                
#                 # Draw a larger circle at the pelvis center position
#                 cv2.circle(annotated_cropped, (pelvis_x, pelvis_y), 8, (0, 255, 0), -1)
                
#                 # Add label with depth info
#                 depth_label = f"PELVIS (z: {pelvis_z:.3f})"
#                 cv2.putText(annotated_cropped, depth_label, (pelvis_x + 10, pelvis_y), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
#                 # Print the coordinates
#                 # print(f"PELVIS CENTER coordinates: {pelvis_data}")

#             # Get lowest keypoint y coordinate
#             lowest_y_global = get_lowest_keypoint_y(results.pose_landmarks, cropped_image.shape[0], y_min) # Call renamed function
#             if lowest_y_global is not None:
#             #      print(f"Lowest Keypoint Y: {lowest_y_global}")
#                  # Optionally draw this point
#                  cv2.circle(annotated_cropped, (cropped_image.shape[1] // 2, lowest_y_global - y_min), 8, (255, 255, 0), -1) # Cyan circle on cropped
#                  cv2.putText(annotated_cropped, "LOWEST_Y", (cropped_image.shape[1] // 2 + 10, lowest_y_global - y_min),
#                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

#         # Place the annotated crop back into the original image
#         annotated_image = image.copy()
#         annotated_image[y_min:y_max, x_min:x_max] = annotated_cropped
        
#         # Draw the bounding box
#         cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
#         # Draw the hybrid pivot point (pelvis_x, bbox_bottom_y)
#         if pelvis_data:
#             pivot_x = pelvis_data[0] # Use pelvis_x_global
#             pivot_y = bbox_bottom_y_global # Use bbox_bottom_y_global
#             cv2.circle(annotated_image, (pivot_x, pivot_y), 8, (255, 0, 255), -1) # Magenta circle
#             cv2.putText(annotated_image, "PIVOT", (pivot_x + 10, pivot_y), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

#         # Draw the lowest y point if found
#         if lowest_y_global is not None and pelvis_data is not None:
#              pivot_x = pelvis_data[0] # Use pelvis x for horizontal position
#              cv2.circle(annotated_image, (pivot_x, lowest_y_global), 8, (255, 255, 0), -1) # Cyan circle
#              # Update label to be more descriptive
#              cv2.putText(annotated_image, "PIVOT (PelvisX, LowestY)", (pivot_x + 10, lowest_y_global),
#                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

#         # Save the annotated image
#         cv2.imwrite(output_path, annotated_image)

#         return annotated_image, pelvis_data, lowest_y_global # Return lowest_y_global

def config_smplx_pred(out):
  smplx_pred = {}
  smplx_pred['global_orient'] = out['smplx_root_pose'].reshape(-1,3).cpu().numpy()
  smplx_pred['body_pose'] = out['smplx_body_pose'].reshape(-1,3).cpu().numpy()
  smplx_pred['left_hand_pose'] = out['smplx_lhand_pose'].reshape(-1,3).cpu().numpy()
  smplx_pred['right_hand_pose'] = out['smplx_rhand_pose'].reshape(-1,3).cpu().numpy()
  smplx_pred['jaw_pose'] = out['smplx_jaw_pose'].reshape(-1,3).cpu().numpy()
  smplx_pred['leye_pose'] = np.zeros((1, 3))
  smplx_pred['reye_pose'] = np.zeros((1, 3))
  smplx_pred['betas'] = out['smplx_shape'].reshape(-1,10).cpu().numpy()
  smplx_pred['expression'] = out['smplx_expr'].reshape(-1,10).cpu().numpy()
  smplx_pred['cam_trans'] = out['cam_trans'].cpu().numpy()
  smplx_pred['root_translation'] = out['root_translation'].cpu().numpy()
  smplx_pred['raw_cam_params'] = out['raw_cam_params'].cpu().numpy()
  smplx_pred['joint_cam'] = out['joint_cam'].reshape(-1,3).cpu().numpy()
  
  # print("out['cam_trans']: ", out['cam_trans'])
  # print("out['root_translation']: ", out['root_translation'])
  
  # print("smplx_pred['joint_cam']: ", smplx_pred['joint_cam'])
  return smplx_pred


def main():
    args = parse_args()
    cudnn.benchmark = True

    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./pretrained_models', args.ckpt_name, 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', args.ckpt_name, f'{args.ckpt_name}.pth.tar')
    img_folder = osp.join(root_dir, 'demo', 'input_frames', args.file_name)
    output_folder = osp.join(root_dir, 'demo', 'output_frames', args.file_name)
    os.makedirs(output_folder, exist_ok=True)
    exp_name = f'inference_{args.file_name}_{args.ckpt_name}_{time_str}'
    # Fix: Save motion files directly in demo/motion folder, not in nested subfolder
    motion_folder = osp.join(root_dir, 'demo', 'motion')

    device = get_device()

    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join(root_dir, 'outputs', exp_name, 'log'),  
            'output_dir': osp.join(root_dir, 'outputs', exp_name),
            'model_dir': osp.join(root_dir, 'outputs', exp_name, 'model_dump'),
            'result_dir': osp.join(root_dir, 'outputs', exp_name, 'result'),
        }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()
    
    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester
    demoer = Tester(cfg)
    demoer.logger.info(f"Using device: {device}")
    demoer.logger.info(f'Inference [{args.file_name}] with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path", 
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)

    start = int(args.start)
    end = int(args.end) + 1

    motion_list = []  # Dictionary to store motion data {frame_index: motion_np_array or None}
    
    for frame in tqdm(range(start, end)):
        # prepare input image
        img_path = osp.join(img_folder, f'{int(frame):06d}.jpg')

        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        original_img_height, original_img_width = original_img.shape[:2]
        
        # detection, xyxy
        yolo_bbox = detector.predict(original_img, 
                                device=get_device(), 
                                classes=00, 
                                conf=cfg.inference.detection.conf, 
                                save=cfg.inference.detection.save, 
                                verbose=cfg.inference.detection.verbose
                                    )[0].boxes.xyxy.detach().cpu().numpy()

        if len(yolo_bbox) < 1:
            # No bounding box detected
            demoer.logger.info(f"Frame {frame}: No bounding box detected!")
            motion_list.append(None)
            continue
            
        # Only select the largest bbox if not multi_person
        if not args.multi_person:
            num_bbox = 1
        else:
            # keep bbox by NMS with iou_thr
            yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)
            num_bbox = len(yolo_bbox)
            
        # ASSUME: only one person in the image
        # loop all detected bboxes
        for bbox_id in range(num_bbox):
            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
            yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
            yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
            yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])
            
            # xywh
            bbox = process_bbox(bbox=yolo_bbox_xywh, 
                                img_width=original_img_width, 
                                img_height=original_img_height, 
                                input_img_shape=cfg.model.input_img_shape, 
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))

            img, trans, inv_trans = generate_patch_image(cvimg=original_img,
                                                bbox=bbox,
                                                scale=1.0,
                                                rot=0.0,
                                                do_flip=False,
                                                out_shape=cfg.model.input_img_shape)
            
            # print(f"Bounding box trans: {trans}, inv_trans: {inv_trans}")
                
            # Create output path for keypoints visualization
            keypoints_output_path = osp.join(output_folder, f'{int(frame):06d}.jpg')
            
            # Get pelvis center and lowest keypoint y from MediaPipe pose detection
            # _, pelvis_center, lowest_keypoint_y = draw_mediapipe_pose(original_img, bbox, keypoints_output_path) # Renamed variable

            img = transform(img.astype(np.float32))/255
            img = img.to(device)[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                    out = demoer.model(inputs, targets, meta_info, 'test')                    
                    smplx_pred = config_smplx_pred(out)
                    
                    # Move pose tensors to the target device immediately after creation
                    global_orient_aa = torch.from_numpy(smplx_pred['global_orient']) # shape (1, 3)
                    body_pose_aa = torch.from_numpy(smplx_pred['body_pose']) # shape (21, 3)
                    left_hand_pose_aa = torch.from_numpy(smplx_pred['left_hand_pose']) # shape (15, 3)
                    right_hand_pose_aa = torch.from_numpy(smplx_pred['right_hand_pose']) # shape (15, 3)
                    
                    # Correction rotation should also be on the device
                    original_global_orient_mat = axis_angle_to_matrix(global_orient_aa)
                    correction_axis_angle = torch.tensor([torch.pi, 0.0, 0.0])
                    correction_rot_mat = axis_angle_to_matrix(correction_axis_angle)
                    corrected_global_orient_mat = correction_rot_mat @ original_global_orient_mat
                    corrected_global_orient_aa = matrix_to_axis_angle(corrected_global_orient_mat)
                    
                    all_poses_aa = torch.cat(
                        [
                          corrected_global_orient_aa,
                          body_pose_aa, 
                          left_hand_pose_aa, 
                          right_hand_pose_aa
                        ],
                        dim=0
                    )
                    
                    all_poses_rotmat = axis_angle_to_matrix(all_poses_aa) # Shape: (52, 3, 3)
                    all_poses_rot6d = matrix_to_rotation_6d(all_poses_rotmat) # Shape: (52, 6)
                    rotations_flat = all_poses_rot6d.reshape(-1) # Shape: (52*6,)
                    
                    # transl = torch.from_numpy(smplx_pred['cam_trans']) # shape (1, 3)
                    
                    
                    # transl = np.array([2, 0, 0]).reshape(1, 3) # Shape: (1, 3)
                    # transl = np.array([0, 2, 0]).reshape(1, 3) # Shape: (1, 3)
                    # transl = np.array([0, 0, 2]).reshape(1, 3) # Shape: (1, 3)
                    
                    # transl = np.array([0, 2, 2]).reshape(1, 3) # Shape: (1, 3)
                    # transl = np.array([2, 0, 2]).reshape(1, 3) # Shape: (1, 3)
                    # transl = np.array([2, 2, 0]).reshape(1, 3) # Shape: (1, 3)
                    
                    # transl = np.array([2, 2, 2]).reshape(1, 3) # Shape: (1, 3)
                    
                    # transl = torch.from_numpy(smplx_pred['raw_cam_params'])
                    transl = torch.from_numpy(smplx_pred['cam_trans'])
                    
                    transl_flat = transl.flatten() # Shape: (3,)
                    
                    # concat to form (3+52*6+3,)
                    motion = np.concatenate((transl_flat, rotations_flat), axis=0)
                    motion_list.append(motion)
    
    # # process the motion list to find missing frames
    missing_frames = [i for i, motion in enumerate(motion_list) if motion is None]
    if len(missing_frames)> 0:
        demoer.logger.info(f"Missing frames detected: {missing_frames}")
        # Interpolate the missing frames
        motion_list = interpolate_missing_motion(motion_list, missing_frames)
        
        # remove all the None values
    # motion_list = [motion for motion in motion_list if motion is not None]

    # Save motion data to npy file
    if motion_list:
        # Create the directory including all intermediate directories if they don't exist
        # Use dirname to get the directory part of the path
        motion_output_path = osp.join(motion_folder, f'{args.file_name}.npy')
        output_dir = osp.dirname(motion_output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the list, translations are relative based on model cam_trans
        np.save(motion_output_path, np.array(motion_list)) 
        demoer.logger.info(f'Saved relative model-based motion data to {motion_output_path}')


if __name__ == "__main__":
    main()
