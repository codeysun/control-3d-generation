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
from threestudio.models.mesh import Mesh
from threestudio.utils.ops import dot

import os
import trimesh
import numpy as np

@threestudio.register("nvdiff-obj-rasterizer")
class NVDiffObjRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        file_path: str = ""
        masked_segments: List[str] = field(default_factory=lambda: [])
        context_type: str = "gl"

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)

        device = get_device()
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, device)

        obj_meshes = []
        control_meshes = []
        self.seg_node_map = {}
        for file in os.listdir(self.cfg.file_path):
            if file.endswith('.obj'):
            # if file in self.cfg.masked_segments:
                obj_file_path = os.path.join(self.cfg.file_path, file)
                mesh = trimesh.load(obj_file_path)
                if mesh.is_empty:
                    print(f"Failed to load OBJ file: {obj_file_path}")
                    continue
                obj_meshes.append(mesh)

                if file in self.cfg.masked_segments:
                    color = [255, 255, 255, 255]
                    control_meshes.append(mesh)
                else:
                    color = [100, 100, 100, 255]
                    # color = [0, 0, 0, 255]
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

        v_pos, t_pos_idx, face_nrm, v_colors = (
            torch.from_numpy(self.trimesh.vertices).float().to(device),
            torch.from_numpy(self.trimesh.faces).long().to(device),
            torch.from_numpy(self.trimesh.face_normals).float().to(device),
            torch.from_numpy(self.trimesh.visual.vertex_colors).int().to(device),
        )  # transform back to torch tensor on CUDA
        self.mesh = Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)
        self.f_nrm = face_nrm
        self.v_colors = v_colors[:, :3].float() / 255.0

        # Material options
        self.ambient_light_color = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32).to(device)
        self.diffuse_light_color = torch.tensor([0.8, 0.3, 0.3], dtype=torch.float32).to(device)

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        # height: int,
        # width: int,
        render_rgb: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]
        mesh = self.mesh

        height = 256
        width = 256

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        out = {"opacity": mask_aa, "mesh": mesh}

        gb_normal = torch.zeros(batch_size, height, width, 3).to(rast)
        gb_normal[mask.squeeze(dim=3)] = self.f_nrm[rast[mask.squeeze(dim=3)][:, 3].int() - 1]
        out.update({"comp_normal": gb_normal})  # in [0, 1]

        if render_rgb:
            selector = mask[..., 0]

            gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
            gb_viewdirs = F.normalize(
                gb_pos - camera_positions[:, None, None, :], dim=-1
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
            gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

            control_mask, _ = self.ctx.interpolate_one(self.v_colors, rast, mesh.t_pos_idx)
            control_mask_aa = self.ctx.antialias(control_mask, rast, v_pos_clip, mesh.t_pos_idx)

            out.update({"comp_rgb": gb_rgb_aa, "comp_rgb_bg": gb_rgb_bg, 
                        "mask": control_mask_aa[..., 0].unsqueeze(-1), "depth": rast[..., 2].unsqueeze(-1)})

        return out

    @property
    def control_mesh(self):
        return self.control_trimesh
