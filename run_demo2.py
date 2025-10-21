
import os
import trimesh
import yaml
import numpy as np
import cv2
import torch

from PIL import Image
from estimater import Any6D

from foundationpose.Utils import get_bounding_box, visualize_frame_results, calculate_chamfer_distance_gt_mesh, align_mesh_to_coordinate
import nvdiffrast.torch as dr
import argparse
from pytorch_lightning import seed_everything

from sam2_instantmesh import *

glctx = dr.RasterizeCudaContext()

def save_img(img, name):
    try:
        img = np.array(img)
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        elif img.dtype == np.uint8:
            img = img[:, :, ::-1]       # BGR to RGB
        elif img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)[:, :, ::-1]

    except:
        if type(img) == Image.Image:
            img = np.array(img)[:, :, ::-1]       # RGB to BGR
        else:
            raise ValueError(f"Unsupported image dtype: {type(img)}")
    img = np.array(img)
    cv2.imwrite(f'./proc_image/{name}.png', img)


if __name__=='__main__':

    seed_everything(0)

    parser = argparse.ArgumentParser(description="Set experiment name and paths")
    parser.add_argument("--ycb_model_path", type=str, default="/home/miruware/ssd_4tb/dataset/ho3d/YCB_Video_Models", help="Path to the YCB Video Models")
    parser.add_argument("--img_to_3d", action="store_true",help="Running with InstantMesh+SAM2")
    args = parser.parse_args()


    ycb_model_path = args.ycb_model_path
    img_to_3d = True #args.img_to_3d

    results = []
    demo_path = 'demo_data'
    mesh_path = os.path.join(demo_path, f'mustard.obj')

    obj = 'demo_mustard'
    save_path = f'results/{obj}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    depth_scale = 1000.0
    color = cv2.cvtColor(cv2.imread(os.path.join(demo_path, 'color.png')), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(demo_path, 'depth.png'), cv2.IMREAD_ANYDEPTH).astype(np.float32) / depth_scale
    Image.fromarray(color).save(os.path.join(save_path, 'color.png'))

    label = np.load(os.path.join(demo_path, 'labels.npz'))
    obj_num = 5
    mask = np.where(label['seg'] == obj_num, 255, 0).astype(np.bool_)
    
    save_img(mask, 'raw_mask')

    img_ls = sorted(os.listdir(f"captured_images"))
    colors = []
    masks = []
    depths = []
    for img in img_ls:
        if img.startswith("color_"):
            colors.append(cv2.imread(f"captured_images/{img}"))
        elif img.startswith("mask_color_"):
            masks.append(cv2.imread(f"captured_images/{img}", cv2.IMREAD_ANYDEPTH).astype(np.bool_))
        elif img.startswith("depth_"):
            depths.append(cv2.imread(f"captured_images/{img}", cv2.IMREAD_ANYDEPTH).astype(np.float32) / depth_scale)

    # Check if we have any masks loaded
    if len(masks) == 0:
        print("Warning: No mask images found!")
        masks = []
    else:
        # Check mask dimensions and resize if necessary
        print(f"Found {len(masks)} mask images")
        if len(masks) > 0:
            # Get the first mask's dimensions as reference
            ref_height, ref_width = masks[0].shape[:2]
            print(f"Reference mask size: {ref_height}x{ref_width}")
            
            # Resize all masks to the same size
            resized_masks = []
            for i, mask in enumerate(masks):
                if mask.shape[:2] != (ref_height, ref_width):
                    print(f"Resizing mask {i} from {mask.shape[:2]} to {ref_height}x{ref_width}")
                    mask = cv2.resize(mask.astype(np.uint8), (ref_width, ref_height), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
                resized_masks.append(mask)
            masks = resized_masks
    
    # Check and resize colors to same size
    if len(colors) == 0:
        print("Warning: No color images found!")
        colors = []
    else:
        print(f"Found {len(colors)} color images")
        if len(colors) > 0:
            # Get the first color's dimensions as reference
            ref_height, ref_width = colors[0].shape[:2]
            print(f"Reference color size: {ref_height}x{ref_width}")
            
            # Resize all colors to the same size
            resized_colors = []
            for i, color in enumerate(colors):
                if color.shape[:2] != (ref_height, ref_width):
                    print(f"Resizing color {i} from {color.shape[:2]} to {ref_height}x{ref_width}")
                    color = cv2.resize(color, (ref_width, ref_height), interpolation=cv2.INTER_LINEAR)
                resized_colors.append(color)
            colors = resized_colors
    
    # Check and resize depths to same size
    if len(depths) == 0:
        print("Warning: No depth images found!")
        depths = []
    else:
        print(f"Found {len(depths)} depth images")
        if len(depths) > 0:
            # Get the first depth's dimensions as reference
            ref_height, ref_width = depths[0].shape[:2]
            print(f"Reference depth size: {ref_height}x{ref_width}")
            
            # Resize all depths to the same size
            resized_depths = []
            for i, depth in enumerate(depths):
                if depth.shape[:2] != (ref_height, ref_width):
                    print(f"Resizing depth {i} from {depth.shape[:2]} to {ref_height}x{ref_width}")
                    depth = cv2.resize(depth, (ref_width, ref_height), interpolation=cv2.INTER_NEAREST)
                resized_depths.append(depth)
            depths = resized_depths
    
    colors = np.array(colors)
    if len(masks) > 0:
        masks = np.array(masks)
    else:
        masks = np.array([])
    depths = np.array(depths)


    input_images = []

    for i in range(len(colors)):
        color = colors[i]
        if len(masks) > 0 and i < len(masks):
            mask = masks[i]
        else:
            print(f"Warning: No mask available for image {i}, skipping...")
            continue
            
        if i < len(depths):
            depth = depths[i]
        else:
            print(f"Warning: No depth available for image {i}, skipping...")
            continue

        if img_to_3d:
            cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
            input_box = np.array([cmin, rmin, cmax, rmax])[None, :]
            # mask_refine = running_sam_box(color, input_box)

            input_image = preprocess_image(color, mask, save_path, ids=i)
            
            # Convert PIL Image to numpy array and normalize to 0-1
            input_image_np = np.array(input_image).astype(np.float32) / 255.0
            
            # Resize to 320x320
            input_image_resized = cv2.resize(input_image_np, (320, 320), interpolation=cv2.INTER_LINEAR)
            input_image_resized = input_image_resized[:, :, :3]
            # Convert from HWC to CHW format [3, 320, 320]
            input_image_chw = np.transpose(input_image_resized, (2, 0, 1))
            
            input_images.append(input_image_chw)

    # Convert list to numpy array [N, 3, 320, 320]
    if len(input_images) > 0:
        input_images = np.array(input_images)
        print(f"Input images shape: {input_images.shape}")
    else:
        print("Warning: No input images processed!")
        input_images = np.array([])

    # if img_to_3d:
    #     cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
    #     input_box = np.array([cmin, rmin, cmax, rmax])[None, :]
    #     mask_refine = running_sam_box(color, input_box)

    #     save_img(color, 'raw_color')
    #     save_img(mask_refine, 'refined_sam_mask')

    #     input_image = preprocess_image(color, mask_refine, save_path, obj)
        
    #     save_img(input_image, 'preprocessed_image')

    #     images = diffusion_image_generation(save_path, save_path, obj, input_image=input_image)

    #     for i, im in enumerate(images):
    #         ii = np.transpose(im, (1, 2, 0))
    #         save_img(ii, f'diffusion_image_{i}')
        
    #     instant_mesh_process(images, save_path, obj)

    #     mesh = trimesh.load(os.path.join(save_path, f'mesh_{obj}.obj'))
    #     mesh = align_mesh_to_coordinate(mesh)
    #     mesh.export(os.path.join(save_path, f'center_mesh_{obj}.obj'))

    #     mesh = trimesh.load(os.path.join(save_path, f'center_mesh_{obj}.obj'))
    # else:
    #     mesh = trimesh.load(mesh_path)
