from typing import Any, Dict
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt
from torch import Tensor

import argparse
import trimesh
import numpy as np
import torch
import sys
import os
import imageio
import torch
import torch.nn.functional as F

import nvdiffrast.torch as dr

def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)

def get_projection_matrix(
    fovy: Float[Tensor, "B"], aspect_wh: float, near: float, far: float
) -> Float[Tensor, "B 4 4"]:
    batch_size = fovy.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32).to(fovy)
    proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / torch.tan(
        fovy / 2.0
    )  # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx

def get_mvp_matrix(
    c2w: Float[Tensor, "B 4 4"], proj_mtx: Float[Tensor, "B 4 4"]
) -> Float[Tensor, "B 4 4"]:
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx

def vertex_transform(
    verts: Float[Tensor, "Nv 3"], mvp_mtx: Float[Tensor, "B 4 4"]
) -> Float[Tensor, "B Nv 4"]:
    verts_homo = torch.cat(
        [verts, torch.ones([verts.shape[0], 1]).to(verts)], dim=-1
    )
    return torch.matmul(verts_homo, mvp_mtx.permute(0, 2, 1))

class ControlRasterizer():
    def __init__(
        self,
        file_path,
        masked_segments,
        device
    ):
        self.ctx = dr.RasterizeGLContext(device=device)

        obj_meshes = []
        control_meshes = []
        self.seg_node_map = {}
        for file in os.listdir(file_path):
            if file.endswith('.obj'):
            # if file in masked_segments:
                obj_file_path = os.path.join(file_path, file)
                mesh = trimesh.load(obj_file_path)
                if mesh.is_empty:
                    print(f"Failed to load OBJ file: {obj_file_path}")
                    continue
                obj_meshes.append(mesh)

                if masked_segments is None or file in masked_segments:
                    color = [255, 255, 255, 255]
                    control_meshes.append(mesh)
                else:
                    # color = [100, 100, 100, 255]
                    color = [0, 0, 0, 255]
                v_colors = np.array([color] * mesh.vertices.shape[0])
                mesh.visual.vertex_colors = v_colors
        
        self.trimesh = trimesh.util.concatenate(obj_meshes)
        self.control_trimesh = trimesh.util.concatenate(control_meshes)

        # Set object pose to align with MVDream
        obj_pose = np.array([
            [0, 0, -1, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        self.trimesh = self.trimesh.apply_transform(obj_pose)
        self.control_trimesh = self.control_trimesh.apply_transform(obj_pose)

        self.v_pos, self.t_pos_idx, self.f_nrm, v_colors = (
            torch.from_numpy(self.trimesh.vertices).float().to(device),
            torch.from_numpy(self.trimesh.faces).long().to(device),
            torch.from_numpy(self.trimesh.face_normals).float().to(device),
            torch.from_numpy(self.trimesh.visual.vertex_colors).int().to(device),
        )  # transform back to torch tensor on CUDA
        self.v_colors = v_colors[:, :3].float() / 255.0

        # Material options
        self.ambient_light_color = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32).to(device)
        self.diffuse_light_color = torch.tensor([0.8, 0.3, 0.3], dtype=torch.float32).to(device)

        # Camera
        fovy_deg: Float[Tensor, "B"] = torch.tensor([40.]).to(device)
        fovy = fovy_deg * np.pi / 180
        self.proj: Float[Tensor, "B 4 4"] = get_projection_matrix(fovy, 1, 0.1, 1000.0)

    def __call__(
        self,
        # mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        # camera_positions: Float[Tensor, "B 3"],
        # light_positions: Float[Tensor, "B 3"],
    ) -> Dict[str, Any]:
        batch_size = c2w.shape[0]

        height = 256
        width = 256

        mvp_mtx = get_mvp_matrix(c2w, self.proj)
        camera_positions: Float[Tensor, "B 3"] = c2w[:, :3, 3]
        light_positions = camera_positions

        v_pos_clip = vertex_transform(self.v_pos, mvp_mtx)

        rast, _ = dr.rasterize(self.ctx, v_pos_clip.float(), self.t_pos_idx.int(), (height, width), grad_db=True)
        mask = rast[..., 3:] > 0
        mask_aa = dr.antialias(mask.float(), rast, v_pos_clip.float(), self.t_pos_idx.int())

        out = {"opacity": mask_aa}

        gb_normal = torch.zeros(batch_size, height, width, 3).to(rast)
        gb_normal[mask.squeeze(dim=3)] = self.f_nrm[rast[mask.squeeze(dim=3)][:, 3].int() - 1]
        out.update({"comp_normal": gb_normal})  # in [0, 1]

        selector = mask[..., 0]

        gb_pos, _ = dr.interpolate(
            self.v_pos.float(), rast, self.t_pos_idx.int(), rast_db=None, diff_attrs=None
        )
        gb_light_positions = light_positions[:, None, None, :].expand(
            -1, height, width, -1
        )

        positions = gb_pos[selector]
        shading_normal = gb_normal[selector]

        light_directions: Float[Tensor, "B ... 3"] = F.normalize(
            gb_light_positions[selector] - positions, dim=-1
        )
        diffuse_light: Float[Tensor, "B ... 3"] = (
            torch.abs(dot(shading_normal, light_directions)) * self.diffuse_light_color
        )
        textureless_color = diffuse_light + self.ambient_light_color

        rgb_fg = textureless_color

        gb_rgb_fg = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
        gb_rgb_fg[selector] = rgb_fg

        gb_rgb_bg = torch.ones(batch_size, height, width, 3).to(rgb_fg)
        gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
        gb_rgb_aa = dr.antialias(gb_rgb.float(), rast, v_pos_clip.float(), self.t_pos_idx.int())

        control_mask, _ = dr.interpolate(
            self.v_colors.float(), rast, self.t_pos_idx.int(), rast_db=None, diff_attrs=None
        )
        control_mask_aa = dr.antialias(control_mask.float(), rast, v_pos_clip.float(), self.t_pos_idx.int())

        out.update({"comp_rgb": gb_rgb_aa, "comp_rgb_bg": gb_rgb_bg, 
                    "mask": control_mask_aa[..., 0].unsqueeze(-1), "depth": rast[..., 2].unsqueeze(-1)})

        return out
