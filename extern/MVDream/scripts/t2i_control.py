import os
import sys
import random
import argparse
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import torch 

from torchvision.transforms.functional import to_pil_image, pil_to_tensor

from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model
from mvdream.annotator.midas import MidasDetector
from mvdream.annotator.util import resize_image, HWC3

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    print(f'Loaded model config from [{config_path}]')
    return model

midas = MidasDetector()

def t2i(model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8, ddim_eta=0., dtype=torch.float32, device="cuda", camera=None, num_frames=1, control=None, control_mask=None):
    if type(prompt)!=list:
        prompt = [prompt]


    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": c.repeat(batch_size,1,1)}
        uc_ = {"context": uc.repeat(batch_size,1,1)}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames
        
        if control is not None:
            c_["control"] = torch.cat([control] * 2)
            c_["effective_region_mask"] = uc_["effective_region_mask"] = control_mask
        else:
            c_["control"] = None
            c_["effective_region_mask"] = uc_["effective_region_mask"] = None

        uc_["control"] = None
        

        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(S=step, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc_,
                                        eta=ddim_eta, x_T=None)
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view", help="load pre-trained model from hugginface")
    parser.add_argument("--config_path", type=str, default=None, help="load model from local config (override model_name)")
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to local checkpoint")
    parser.add_argument("--text", type=str, default="an astronaut riding a horse")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=4, help="num of frames (views) to generate")
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=15)
    parser.add_argument("--camera_azim", type=int, default=90)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    batch_size = max(4, args.num_frames)

    print("load t2i model ... ")
    model_ckpt = torch.load(args.ckpt_path)
    model = create_model(args.model_name)
    model.load_state_dict(model_ckpt['state_dict'], strict=False)
    
    model.to(device)
    model.eval()

    sampler = DDIMSampler(model)
    uc = model.get_learned_conditioning( [""] ).to(device)
    print("load t2i model done . ")

    # pre-compute camera matrices
    if args.use_camera:
        camera = get_camera(args.num_frames, elevation=args.camera_elev, 
                azimuth_start=args.camera_azim, azimuth_span=args.camera_azim_span)
        camera = camera.repeat(batch_size//args.num_frames,1).to(device)
    else:
        camera = None

    # get control images
    control_rgb = []
    for i in range(4):
        # image_path = f"imgs/iman5000-{i}.png"
        image_path = f"imgs/mug5000-{i}.png"
        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        control_rgb.append(image)

    # Control_rgb must be shape BCHW
    detected_maps = []
    for i in range(len(control_rgb)):
        img_detect = np.array(control_rgb[i])  
        # depth control
        detected_map, _ = midas(img_detect, 100,200) # MiDas Depth

        detected_map = torch.from_numpy(HWC3(detected_map).copy()).permute(2,0,1).float().cuda() / 255.0
        detected_maps.append(detected_map)
    input_cond = torch.stack(detected_maps)

    control_mask = []
    for i in range(4):
        # image_path = f"imgs/iman5000-{i}mask.png"
        image_path = f"imgs/mug5000-{i}mask.png"
        # image_path = f"imgs/halfmask.png"
        image = Image.open(image_path).convert('L')
        image = image.resize((256, 256))
        image = pil_to_tensor(image) / 255.0
        control_mask.append(image)
    control_mask = torch.stack(control_mask)



    # input_cond = None
    # control_mask = None
    
    t = args.text + args.suffix
    set_seed(args.seed)
    images = []
    for j in range(3):
        img = t2i(model, args.size, t, uc, sampler, step=50, scale=10, batch_size=batch_size, ddim_eta=0.0, 
                dtype=dtype, device=device, camera=camera, num_frames=args.num_frames, control=input_cond,
                control_mask=control_mask)
        img = np.concatenate(img, 1)
        images.append(img)
    images = np.concatenate(images, 0)
    Image.fromarray(images).save(f"sample.png")