# Training loop for TensoRF from Nerfstudio library.

# Dependencies utilize Nerfstudio APIs and libraries
# Goal is to explicate inner workings of Nerfstudio abstraction


import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import imageio.v2 as imageio
import nerfacc
import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from fabric.utils.event import EventStorage, get_event_storage
from PIL import Image
from torch.nn.parameter import Parameter
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.tensorf_field import TensoRFField
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)

WHITE = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x.device)


def get_normalized_directions(directions: TensorType["bs":..., 3]) -> TensorType["bs":..., 3]:
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


class TCNNInstantNGPField(Field):
    """TCNN implementation of the Instant-NGP field.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_layers_color: number of hidden layers for color network
        hidden_dim_color: dimension of hidden layers for color network
        use_appearance_embedding: whether to use appearance embedding
        num_images: number of images, required if use_appearance_embedding is True
        appearance_embedding_dim: dimension of appearance embedding
        contraction_type: type of contraction
        num_levels: number of levels of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
    """

    def __init__(
        self,
        aabb: TensorType,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        use_appearance_embedding: Optional[bool] = False,
        num_images: Optional[int] = None,
        appearance_embedding_dim: int = 32,
        contraction_type: ContractionType = ContractionType.UN_BOUNDED_SPHERE,
        num_levels: int = 16,
        log2_hashmap_size: int = 19,
        max_res: int = 2048,
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim
        self.contraction_type = contraction_type

        self.use_appearance_embedding = use_appearance_embedding
        if use_appearance_embedding:
            assert num_images is not None
            self.appearance_embedding_dim = appearance_embedding_dim
            self.appearance_embedding = Embedding(num_images, appearance_embedding_dim)

        # TODO: set this properly based on the aabb
        base_res: int = 16
        per_level_scale = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=1 + self.geo_feat_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        in_dim = self.direction_encoding.n_output_dims + self.geo_feat_dim
        if self.use_appearance_embedding:
            in_dim += self.appearance_embedding_dim
        self.mlp_head = tcnn.Network(
            n_input_dims=in_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        positions = ray_samples.frustums.get_positions()
        positions_flat = positions.view(-1, 3)
        positions_flat = contract(x=positions_flat, roi=self.aabb, type=self.contraction_type)

        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)

        d = self.direction_encoding(directions_flat)
        if density_embedding is None:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
            h = torch.cat([d, positions.view(-1, 3)], dim=-1)
        else:
            h = torch.cat([d, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

        if self.use_appearance_embedding:
            if ray_samples.camera_indices is None:
                raise AttributeError("Camera indices are not provided.")
            camera_indices = ray_samples.camera_indices.squeeze()
            if self.training:
                embedded_appearance = self.appearance_embedding(camera_indices)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )
            h = torch.cat([h, embedded_appearance.view(-1, self.appearance_embedding_dim)], dim=-1)

        rgb = self.mlp_head(h).view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions)
        return {FieldHeadNames.RGB: rgb}

    def get_opacity(self, positions: TensorType["bs":..., 3], step_size) -> TensorType["bs":..., 1]:
        """Returns the opacity for a position. Used primarily by the occupancy grid.

        Args:
            positions: the positions to evaluate the opacity at.
            step_size: the step size to use for the opacity evaluation.
        """
        density = self.density_fn(positions)
        ## TODO: We should scale step size based on the distortion. Currently it uses too much memory.
        # aabb_min, aabb_max = self.aabb[0], self.aabb[1]
        # if self.contraction_type is not ContractionType.AABB:
        #     x = (positions - aabb_min) / (aabb_max - aabb_min)
        #     x = x * 2 - 1  # aabb is at [-1, 1]
        #     mag = x.norm(dim=-1, keepdim=True)
        #     mask = mag.squeeze(-1) > 1

        #     dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (1 / mag**3 - (2 * mag - 1) / mag**4)
        #     dev[~mask] = 1.0
        #     dev = torch.clamp(dev, min=1e-6)
        #     step_size = step_size / dev.norm(dim=-1, keepdim=True)
        # else:
        #     step_size = step_size * (aabb_max - aabb_min)

        opacity = density * step_size
        return opacity


class NGPModel(Model):
    """Instant NGP model

    Args:
        config: instant NGP configuration to instantiate model
    """

    config: InstantNGPModelConfig
    field: TCNNInstantNGPField

    def __init__(self, config: InstantNGPModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.field = TCNNInstantNGPField(
            aabb=self.scene_box.aabb,
            contraction_type=self.config.contraction_type,
            use_appearance_embedding=self.config.use_appearance_embedding,
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
        )

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        # Occupancy Grid
        self.occupancy_grid = nerfacc.OccupancyGrid(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            contraction_type=self.config.contraction_type,
        )

        # Sampler
        vol_sampler_aabb = self.scene_box.aabb if self.config.contraction_type == ContractionType.AABB else None
        self.sampler = VolumetricSampler(
            scene_aabb=vol_sampler_aabb,
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            # TODO: needs to get access to the sampler, on how the step size is determinated at each x. See
            # https://github.com/KAIR-BAIR/nerfacc/blob/127223b11401125a9fce5ce269bb0546ee4de6e8/examples/train_ngp_nerf.py#L190-L213
            self.occupancy_grid.every_n_step(
                step=step,
                occ_eval_fn=lambda x: self.field.get_opacity(x, self.config.render_step_size),
            )

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        assert self.field is not None
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                cone_angle=self.config.cone_angle,
            )

        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            packed_info=packed_info,
            sigmas=field_outputs[FieldHeadNames.DENSITY],
            t_starts=ray_samples.frustums.starts,
            t_ends=ray_samples.frustums.ends,
        )

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        alive_ray_mask = accumulation.squeeze(-1) > 0

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "alive_ray_mask": alive_ray_mask,  # the rays we kept from sampler
            "num_samples_per_ray": packed_info[:, 1],
        }
        return outputs


# blender dataset input
# def parse_blender_data(root, split):
#     path = Path(root) / f'transforms_{split}.json'
#     meta = json.load(open(path, encoding='UTF-8'))
#     images = []
#     for frame in meta['frames']:
#         fname = path.parent / Path(frame['file_path'].replace('./', '') + '.png')
#         images.append(fname)


def fov_to_f(h, w, fov):
    # fov on the x axis
    f = (w / 2.) / np.tan(fov / 2.)
    fx = fy = f
    cx = w / 2.
    cy = h / 2.
    return fx, fy, cx, cy


def read_blender_data(root, split, size=256):
    json_path = os.path.join(root, f'transforms_{split}.json')
    camera_json = json.load(open(json_path))
    fov = camera_json['camera_angle_x']
    camera_info = camera_json['frames']

    images = []
    c2ws = []
    for frame in camera_info:
        path = frame['file_path']
        path = os.path.join(root, path + '.png')
        images.append(Image.open(path).resize((size, size)))    # resize image
        c2ws.append(np.array(frame['transform_matrix'])[:3])\

    c2ws = torch.from_numpy(np.stack(c2ws, 0)).float()
    width = images[0].width
    height = images[0].height

    fx, fy, cx, cy = fov_to_f(h=height, w=width, fov=fov)
    fx = torch.from_numpy(np.array(fx)).view(1, 1).float()
    fy = torch.from_numpy(np.array(fy)).view(1, 1).float()
    cx = torch.from_numpy(np.array(cx)).view(1, 1).float()
    cy = torch.from_numpy(np.array(cy)).view(1, 1).float()

    cameras = Cameras(
        camera_to_worlds=c2ws,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height
    )

    radius = cameras.camera_to_worlds[:, :, -1].norm(p=2, dim=-1)
    radius = radius.max().item()

    near = 2.                       # set to 2. for blender
    far = radius * 2. - near        # set to 6. for blender

    images = np.stack([np.array(x) for x in images], 0)
    images = torch.from_numpy(images).float() / 255.
    images = images[...,:3] * images[...,-1:] + (1. - images[...,-1:])  # alpha blending
    # Image.fromarray((images[0] * 255).numpy().astype(np.uint8)).save('hmm.png')
    return images, cameras, near, far, radius


def generate_rays(cameras, far, near):
    rays = cameras.generate_rays(camera_indices=torch.arange(len(cameras)).unsqueeze(-1))
    rays = RayBundle(origins=rays.origins, directions=rays.directions, pixel_area=rays.pixel_area, camera_indices=rays.camera_indices, nears=near, fars=far, metadata=rays.metadata, times=rays.times)
    return rays


def main(
    data_dir: str='/share/data/pals/jjahn/data/blender/lego',
    scene_box: SceneBox=SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32)),

):
    images, cameras, near, far, radius = read_blender_data(data_dir, split='train', size=800)
    test_images, test_cameras, _, _, _ = read_blender_data(data_dir, split='test', size=800)
    assert images.shape[-1] == 3 and len(images) == len(cameras)
    pixels = images.permute(1, 2, 0, 3).contiguous().view(-1, 3) # [N, 3]
    rays = generate_rays(cameras, far, near)
    rays = rays.flatten()

    model = (
        NGPModel(
            scene_box=scene_box
        )
        .to('cuda')
        .train()
    )

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9,0.99))

    num_iter = 30000
    num_rays = len(rays)
    batch_size = 1024
    with EventStorage() as metric:
        for iter_idx in range(num_iter):
            # sample_rays = rays[i * batch_size : (i + 1) * batch_size]
            # sample_targets = pixels[i * batch_size : (i + 1) * batch_size]
            indices = torch.randint(0, num_rays, (batch_size,))
            sample_rays = rays[indices].to('cuda')
            sample_targets = pixels[indices].to('cuda')

            outputs = model.get_outputs(sample_rays)
            loss = F.mse_loss(outputs['rgb'], sample_targets)
            psnr = mse2psnr(loss)
            optimizer.zero_grad()                   # zero out the gradients
            loss.backward()                         # compute the gradients
            optimizer.step()                        # update the parameters
            metric.put_scalars(loss=loss.item(), psnr=psnr.item())
            metric.step()

            # reinitialize the optimizer if the model has changed
            if model.get_training_callbacks(iter_idx):
                optimizer = torch.optim.Adam(model.parameters(), betas=(0.9,0.99))
            # if iter_idx % 100 == 0:
            #     print(f'iter {iter_idx} psnr {psnr.item()}')


if __name__ == '__main__':
    tyro.cli(main)  # tyro is a wrapper around hydra
