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

class MVDreamControlPipeline():
    def __init__(
        self,
        model_name,
        ckpt_path,
        device,
        dtype=torch.float16,

    ):
        self.dtype = dtype
        self.device = device

        model_ckpt = torch.load(ckpt_path)
        self.model = create_model(model_name)
        self.model.load_state_dict(model_ckpt['state_dict'], strict=False)
        
        self.model.to(self.device)
        self.model.eval()

        self.sampler = DDIMSampler(self.model)

    def __call__(self, image_size, prompt, step=20, scale=7.5, batch_size=8, ddim_eta=0., camera=None, num_frames=1, control=None, control_mask=None):
        if type(prompt)!=list:
            prompt = [prompt]

        uc = self.model.get_learned_conditioning( [""] ).to(self.device)

        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=self.dtype):
            c = self.model.get_learned_conditioning(prompt).to(self.device)
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
            samples_ddim, _ = self.sampler.sample(S=step, conditioning=c_,
                                            batch_size=batch_size, shape=shape,
                                            verbose=False, 
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc_,
                                            eta=ddim_eta, x_T=None)
            x_sample = self.model.decode_first_stage(samples_ddim)
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()

        return list(x_sample.astype(np.uint8))
