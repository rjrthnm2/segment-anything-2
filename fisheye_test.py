import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import io
import re 
import subprocess
import csv

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major <= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# sam2_checkpoint = r"D:\Github\segment-anything-2\checkpoints\sam2_hiera_large.pt"
# sam2_checkpoint = r"D:\Github\segment-anything-2\checkpoints\sam2_hiera_base_plus.pt"
sam2_checkpoint = r"D:\Github\segment-anything-2\checkpoints\sam2_hiera_tiny.pt"

# model_cfg = "sam2_hiera_l.yaml"
# model_cfg = "sam2_hiera_b+.yaml"
model_cfg = "sam2_hiera_t.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# mask function
def show_mask(mask, ax, obj_id = None, random_color = False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])],axis = 0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3],0.6])
    h, w = mask.shape [-2:]
    mask_image = mask.reshape(h,w,1)* color.reshape(1,1,-1)
    ax.imshow(mask_image)

# function for showing points
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', alpha=0.5, s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', alpha=0.5, s=marker_size, edgecolor='white', linewidth=1.25)


# Define the video file path and output directory
video_path = r"E:\CSTEPS\Spring_2019\GroupVideos_Fisheye\mp4\012819_CSTEPS2_AD1_TA.mp4"
video_dir = r"E:\CSTEPS\Spring_2019\GroupVideos_Fisheye\TrackingVideo\012819_CSTEPS2_AD1_TA"

# # Create the output directory if it doesn't exist
# if not os.path.exists(video_dir):
#     os.makedirs(video_dir)

# # FFmpeg command to extract images with zero-padded filenames (e.g., 00001.jpg)
# ffmpeg_command = [
#     'ffmpeg',
#     '-i', video_path,
#     '-t', '00:02:00',          # Extract frames from the first 2 minutes
#     '-vf', 'fps=9',            # Set the frame rate to 9 fps (source fps)
#     '-q:v', '2',               # High quality for JPG output
#     os.path.join(video_dir, '%05d.jpg')  # Output filename pattern with zero padding
# ]

# # Run the FFmpeg command using subprocess
# try:
#     subprocess.run(ffmpeg_command, check=True)
#     print(f"Images extracted and saved in {video_dir}")
# except subprocess.CalledProcessError as e:
#     print(f"An error occurred: {e}")

# print("Press Enter to continue...")
# input()
# print("Continuing...")

#scan all the jpg frame names in this directory

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

frame_idx = 0
# plt.figure(figsize=(12,8))
# plt.title(f"Frame {frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
# plt.show()

# print("Press Enter to continue...")
# input()
# print("Continuing...")

inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

ann_frame_idx = 0 # the frame index we interact with
ann_obj_id = 1 # give a unique id to each object we interact with(it can be any integer)

points = np.array([[985,115],[1029,275]], dtype=np.float32)

# for labels, "1" means positive click, "0" means negative click

labels = np.array([1,1], dtype=np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# # show the results on the current (interacted) frame
# plt.figure(figsize=(12,8))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_points(points, labels, plt.gca())
# show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
# plt.show()

# print("Press Enter to continue...")
# input()
# print("Continuing...")

# run propagation throughout the video and collect the results in a dictionary
video_segments = {}

# TODO: need to fix this

with open('output.csv', 'w', newline='') as csvfile:
    fieldnames = ['frame_idx', 'tracked', 'x_min', 'y_min', 'x_max', 'y_max']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            if np.any(mask):
                # Calculate bounding box coordinates
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]

                writer.writerow({
                    'frame_idx': out_frame_idx,
                    'tracked': 1,
                    'x_min': cmin,
                    'y_min': rmin,
                    'x_max': cmax,
                    'y_max': rmax,
                })
            else:
                writer.writerow({
                    'frame_idx': out_frame_idx,
                    'tracked': 0,
                    'x_min': None,
                    'y_min': None,
                    'x_max': None,
                    'y_max': None,
                })

# for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#     video_segments[out_frame_idx] = {
#         out_obj_id:(out_mask_logits[i] > 0.0).cpu().numpy()
#         for i, out_obj_id in enumerate(out_obj_ids)
#     }

# # render the segmentation results every few frames
# vis_frame_stride = 1
# plt.close("all")

# # define the figure outside of the loop
# fig = plt.figure(figsize=(6,4))
# for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
#     plt.title(f"Frame {out_frame_idx}")
#     im=plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])), animated=True)
#     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
#         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    
#     plt.savefig(f'E:/CSTEPS/Spring_2019/GroupVideos_Fisheye/TrackingVideo/012819_CSTEPS2_AD1_TA_output/s{out_frame_idx}.png')


