from dataclasses import dataclass, field

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *

import os
import trimesh
import pyrender
import numpy as np
import cv2

os.environ["PYOPENGL_PLATFORM"] = "egl"

@threestudio.register("pyrender-rasterizer")
class PyRenderRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        file_path: str = ""
        masked_segments: List[str] = field(default_factory=lambda: [])

    cfg: Config

    def configure(
        self,
    ) -> None:
        # Set up pyrender scene and offscreenrenderer
        self.scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])
        self.renderer = pyrender.OffscreenRenderer(0, 0)

        # Set material
        material_pr = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0)
        )

        # Set object pose to align with MVDream
        obj_pose = np.array([
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])


        # Load obj files and add to scenes
        self.obj_meshes = []
        self.seg_node_map = {}
        for file in os.listdir(self.cfg.file_path):
            if file.endswith('.obj'):
            # if file in self.cfg.masked_segments:
                obj_file_path = os.path.join(self.cfg.file_path, file)
                mesh = trimesh.load(obj_file_path)
                if mesh.is_empty:
                    print(f"Failed to load OBJ file: {obj_file_path}")
                    continue
                self.obj_meshes.append(mesh)
                mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=False, wireframe=False, material=material_pr)

                obj_node = self.scene.add(mesh_pr, pose=obj_pose)
                if file in self.cfg.masked_segments:
                    self.seg_node_map[obj_node] = [255, 255, 255]
                # TODO: set up weak control for other parts
                else:
                    self.seg_node_map[obj_node] = [100, 100, 100]

    @staticmethod
    def convert_blender_to_opengl(camera_matrix):
        if isinstance(camera_matrix, np.ndarray):
            # Construct transformation matrix to convert from OpenGL space to Blender space
            flip_yz = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            camera_matrix_opengl = np.dot(flip_yz, camera_matrix)
        else:
            # Construct transformation matrix to convert from OpenGL space to Blender space
            flip_yz = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            if camera_matrix.ndim == 3:
                flip_yz = flip_yz.unsqueeze(0)
            camera_matrix_opengl = torch.matmul(flip_yz.to(camera_matrix), camera_matrix)
        return camera_matrix_opengl

    @staticmethod
    def dilate_and_feather_mask(mask):
        H, W = mask.shape

        # Kernel sizes are found empirically on 64x64 image
        # Make kernel size scale with image dimensions
        scale = H / 64

        # New kernel sizes
        dilation_kernel_size = int(np.ceil(1 * scale))
        blur_kernel_size = int(np.ceil(5 * scale))

        # Ensure that kernel sizes are odd numbers (as required by many functions)
        if dilation_kernel_size % 2 == 0:
            dilation_kernel_size += 1
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1


        dilation_kernel_size = (dilation_kernel_size, dilation_kernel_size)
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel_size)
    
        dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=1)

        # feather_kernel_size = (blur_kernel_size, blur_kernel_size)
        # feathered_mask = cv2.GaussianBlur(dilated_mask, feather_kernel_size, 1.0)

        # return feathered_mask
        return dilated_mask

    def forward(
        self,
        fovy: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        **kwargs
    ) -> Dict[str, Any]:
        device = fovy.device

        batch_size = fovy.shape[0]

        # self.renderer.viewport_width = width
        # self.renderer.viewport_height = height
        self.renderer.viewport_width = 256
        self.renderer.viewport_height = 256

        # Convert tensors to numpy and convert cooordinate systems
        fovy_rad = fovy.cpu() * np.pi / 180
        c2w_np = PyRenderRasterizer.convert_blender_to_opengl(c2w).cpu()

        flip_yz = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).to(light_positions)
        light_positions_np = (torch.matmul(flip_yz, light_positions.T)).T.cpu()

        # No way to batch render with pyrender, so just iterate
        mask_list = []
        depth_list = []
        rgb_list = []

        for i in range(batch_size):
            # Add camera and light
            camera = pyrender.PerspectiveCamera(yfov=fovy_rad[i])
            camera_node = self.scene.add(camera, pose=c2w_np[i])
            plight = pyrender.PointLight(color=np.array([0.0, 1.0, 0.0]), intensity=5.0)
            plight_node = self.scene.add(plight, pose=np.array([[1, 0, 0, light_positions_np[i, 0]],
                                                                [0, 1, 0, light_positions_np[i, 1]],
                                                                [0, 0, 1, light_positions_np[i, 2]],
                                                                [0, 0, 0, 1]]))

            rgb, depth = self.renderer.render(self.scene)
            mask, _ = self.renderer.render(self.scene, flags=pyrender.RenderFlags.SEG, seg_node_map=self.seg_node_map)

            rgb_list.append(torch.from_numpy(rgb.copy()))
            depth_list.append(torch.from_numpy(depth.copy()).unsqueeze(-1))

            # Dilate and feather the mask
            # mask = PyRenderRasterizer.dilate_and_feather_mask(mask[:, :, 0]) # only take R channel
            mask_tensor = torch.from_numpy(mask[:, :, 0].copy())
            mask_tensor = mask_tensor.unsqueeze(-1)
            mask_list.append(mask_tensor)

            self.scene.remove_node(camera_node)
            self.scene.remove_node(plight_node)

        comp_rgb: Float[Tensor, "B H W 3"] = torch.stack(rgb_list).to(device)
        comp_rgb = comp_rgb.float() / 255.0
        depths: Float[Tensor, "B H W 1"] = torch.stack(depth_list).to(device)
        depths = depths.float() / 255.0
        masks: Float[Tensor, "B H W 1"] = torch.stack(mask_list).to(device)
        masks = masks.float() / 255.0

        out = {
            "comp_rgb": comp_rgb,
            "depth": depths,
            "mask": masks,
        }

        return out
