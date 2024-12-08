import os
import tyro
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
import gradio as gr
import kornia

import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

from extern.LGM.core.options import AllConfigs, Options
from extern.LGM.core.models import LGM

from extern.MVDream.mvdream.camera_utils import get_camera
from extern.MVDream.scripts.control_renderer import ControlRasterizer
from extern.MVDream.mvdream.annotator.midas import MidasDetector
from extern.MVDream.scripts.t2i_control_pipeline import MVDreamControlPipeline
from extern.MVDream.mvdream.annotator.util import resize_image, HWC3

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
GRADIO_VIDEO_PATH = 'gradio_output.mp4'
GRADIO_PLY_PATH = 'gradio_output.ply'

opt = tyro.cli(AllConfigs)

# model
model = LGM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    resume = "extern/LGM/pretrained/model_fp16.safetensors"
    ckpt = load_file(resume, device='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {resume}')



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().to(device)
model.eval()

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

mv_ckpt_path = "./extern/MVDream/mvdream/ckpt/depth_model.ckpt"
mv_model_name = "./extern/MVDream/mvdream/configs/cldm_v21_mvdream.yaml"

pipe_control = MVDreamControlPipeline(mv_model_name, mv_ckpt_path, device=device.type, dtype=torch.float16)

midas = MidasDetector()

# load rembg
bg_remover = rembg.new_session()

def get_camera_nvdiffrast(n_views, eval_elevation_deg, eval_camera_distance):
    azimuth_deg = torch.linspace(0, 360.0, n_views+1)[:n_views]
    elevation_deg = torch.full_like(
        azimuth_deg, eval_elevation_deg
    )
    camera_distances = torch.full_like(
        elevation_deg, eval_camera_distance
    )

    elevation = elevation_deg * np.pi / 180
    azimuth = azimuth_deg * np.pi / 180

    # convert spherical coordinates to cartesian coordinates
    # right hand coordinate system, x back, y right, z up
    # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )

    # default scene center at origin
    center = torch.zeros_like(camera_positions)
    # default camera up direction as +z
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
        None, :
    ].repeat(1, 1)

    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat(
        [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
    )
    c2w[:, 3, 3] = 1.0
    return c2w

def dilate_and_feather_mask(mask):
    B, C, H, W = mask.shape

    # Kernel sizes are found empirically on 64x64 image
    # Make kernel size scale with image dimensions
    scale = H / 64

    # New kernel sizes
    dilation_kernel_size = int(np.ceil(5 * scale))

    # Ensure that kernel sizes are odd numbers (as required by many functions)
    if dilation_kernel_size % 2 == 0:
        dilation_kernel_size += 1

    dilation_kernel = torch.ones(dilation_kernel_size, dilation_kernel_size).to(mask)
    dilated_mask = kornia.morphology.dilation(mask, dilation_kernel)

    return dilated_mask

# process function
def process(prompt, guidance_scale, prompt_neg='', input_elevation=0, input_num_steps=30, input_seed=42):

    # seed
    kiui.seed_everything(input_seed)

    os.makedirs(opt.workspace, exist_ok=True)
    output_video_path = os.path.join(opt.workspace, GRADIO_VIDEO_PATH)
    output_ply_path = os.path.join(opt.workspace, GRADIO_PLY_PATH)

    image_size = 256
    c2ws = get_camera_nvdiffrast(4, 0.0, 3.0).to(device)

    control_obj_path = "/home/codeysun/git/data/PartNet/data_v0/8677/objs"
    control_masked_segments = ["original-3.obj"]

    input_cond = None
    control_mask = None
    condition_image_grid = None
    if control_obj_path is not None:
        # Process control signal
        control_rasterizer = ControlRasterizer(control_obj_path, control_masked_segments, device)
        control_dict = control_rasterizer(c2ws)

        control_rgb_BCHW = control_dict["comp_rgb"].permute(0, 3, 1, 2)
        control_rgb = F.interpolate(control_rgb_BCHW, (image_size, image_size), mode='bilinear', align_corners=False)
        control_mask: Float[Tensor, "B 1 H W"] = control_dict["mask"].permute(0, 3, 1, 2)
        control_mask = dilate_and_feather_mask(control_mask)

        # Apply MiDaS to control image
        detected_maps = []
        condition_image = []
        for i in range(len(control_rgb)):
            img_load= TF.to_pil_image(control_rgb[i].detach().cpu())
            img_detect = np.array(img_load)  

            # depth control
            detected_map, _ = midas(img_detect, 100,200) # MiDas Depth

            detected_map_hwc = HWC3(detected_map)

            mask = control_mask[i].detach().cpu().numpy().transpose(1, 2, 0)
            mask = (mask * 0.9) + 0.1
            depth_masked = (detected_map_hwc.astype(np.float32) / 255.0) * mask
            # depth_masked = detected_map_hwc

            condition_image.append(depth_masked)
            # condition_image.append(img_detect.astype(np.float32) / 255.0)

            detected_map_torch = torch.from_numpy(detected_map_hwc.copy()).permute(2,0,1).float().to(device) / 255.0
            detected_maps.append(detected_map_torch)
        input_cond = torch.stack(detected_maps).to(device)

        condition_image_grid = np.concatenate([
            np.concatenate([condition_image[1], condition_image[2]], axis=1),
            np.concatenate([condition_image[3], condition_image[0]], axis=1),
        ], axis=0)


    # Get MVDream image conditioned on input condition
    # TODO: add negative prompts?
    c2ws_pipe = get_camera(4, elevation=0, azimuth_start=90)
    c2ws_pipe = c2ws_pipe.repeat(1,1).to(device)
    pipe_control.model.control_scales = [guidance_scale] * 13
    mv_image_uint8 = pipe_control(image_size, prompt, step=50, scale=10, batch_size=4, ddim_eta=0.0, camera=c2ws_pipe, num_frames=4,
                                  control=input_cond, control_mask=control_mask)

    # remove BG
    mv_image = []
    for i in range(4):
        image = rembg.remove(mv_image_uint8[i], session=bg_remover) # [H, W, 4]
        # to white bg
        image = image.astype(np.float32) / 255
        # image = recenter(image, image[..., 0] > 0, border_ratio=0.2) # [H, W, 4]
        image = image[..., :3] * image[..., -1:] + (1 - image[..., -1:])
        mv_image.append(image)

    mv_image_grid = np.concatenate([
        np.concatenate([mv_image[1], mv_image[2]], axis=1),
        np.concatenate([mv_image[3], mv_image[0]], axis=1),
    ], axis=0)

    # generate gaussians
    input_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32
    input_image = torch.from_numpy(input_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    rays_embeddings = model.prepare_default_rays(device, elevation=input_elevation)
    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)
        
        # save gaussians
        model.gs.save_ply(gaussians, output_ply_path)
        
        # render 360 video 
        images = []
        elevation = 0
        if opt.fancy_video:
            azimuth = np.arange(0, 720, 4, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                scale = min(azi / 360, 1)

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
        else:
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.concatenate(images, axis=0)
        imageio.mimwrite(output_video_path, images, fps=30)

    return mv_image_grid, output_video_path, output_ply_path, condition_image_grid

# gradio UI

_TITLE = '''LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation'''

_DESCRIPTION = '''
<div>
<a style="display:inline-block" href="https://me.kiui.moe/lgm/"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
<a style="display:inline-block; margin-left: .5em" href="https://github.com/3DTopia/LGM"><img src='https://img.shields.io/github/stars/3DTopia/LGM?style=social'/></a>
</div>

* Input can be only text, only image, or both image and text. 
* If you find the output unsatisfying, try using different seeds!
'''

block = gr.Blocks(title=_TITLE).queue()
with block:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('# ' + _TITLE)
    gr.Markdown(_DESCRIPTION)
    
    with gr.Row(variant='panel'):
        with gr.Column(scale=1):
            # input prompt
            input_text = gr.Textbox(label="prompt")
            # negative prompt
            input_neg_text = gr.Textbox(label="negative prompt", value='ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate')
            # guidance scale
            guidance_scale = gr.Slider(label="guidance scale", minimum=0, maximum=10, step=0.1, value=1.0)
            # elevation
            input_elevation = gr.Slider(label="elevation", minimum=-90, maximum=90, step=1, value=0)
            # inference steps
            input_num_steps = gr.Slider(label="inference steps", minimum=1, maximum=100, step=1, value=30)
            # random seed
            input_seed = gr.Slider(label="random seed", minimum=0, maximum=100000, step=1, value=0)
            # gen button
            button_gen = gr.Button("Generate")

        
        with gr.Column(scale=1):
            with gr.Tab("Video"):
                # final video results
                output_video = gr.Video(label="video")
                # ply file
                output_file = gr.File(label="ply")
            with gr.Tab("Multi-view Image"):
                # multi-view results
                output_image = gr.Image(interactive=False, show_label=False)
            with gr.Tab("Input Condition"):
                output_condition = gr.Image(interactive=False, show_label=False)
        
        inputs = [input_text, guidance_scale, input_neg_text, input_elevation, input_num_steps, input_seed]
        button_gen.click(process, inputs=inputs, outputs=[output_image, output_video, output_file, output_condition])
    #
    # gr.Examples(
    #     examples=[
    #         "data_test/anya_rgba.png",
    #         "data_test/bird_rgba.png",
    #         "data_test/catstatue_rgba.png",
    #     ],
    #     inputs=[input_image],
    #     outputs=[output_image, output_video, output_file],
    #     fn=lambda x: process(input_image=x, prompt=''),
    #     cache_examples=False,
    #     label='Image-to-3D Examples'
    # )
    #
    # gr.Examples(
    #     examples=[
    #         "a motorbike",
    #         "a hamburger",
    #         "a furry red fox head",
    #     ],
    #     inputs=[input_text],
    #     outputs=[output_image, output_video, output_file],
    #     fn=lambda x: process(input_image=None, prompt=x),
    #     cache_examples=False,
    #     label='Text-to-3D Examples'
    # )
    
block.launch(server_name="0.0.0.0", share=True)
