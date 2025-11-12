# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from enum import IntEnum
import glob
import importlib
import sys

import torch
import torch.utils.cpp_extension
from torch.utils.cpp_extension import get_default_build_root

from threedgrut.datasets.protocols import Batch
from threedgrut.utils.timer import CudaTimer

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
#

def _load_from_torch_cache(mod_name: str = "lib3dgrt_cc"):
    """Locate the compiled .so in the torch extensions cache, import it,
    and register it under both 'lib3dgrt_cc' and 'threedgrt_tracer.lib3dgrt_cc'."""
    root = os.environ.get("TORCH_EXTENSIONS_DIR", get_default_build_root())
    matches = glob.glob(
        os.path.join(root, "py*", mod_name, "**", f"{mod_name}*.so"),
        recursive=True,
    )
    if not matches:
        raise ImportError(f"Built extension {mod_name} not found in torch cache: {root}")
    # pick the newest artifact in case there are multiple builds
    so_path = max(matches, key=os.path.getmtime)

    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # Make both import styles work elsewhere in the codebase
    sys.modules[mod_name] = mod
    sys.modules[f"threedgrt_tracer.{mod_name}"] = mod
    return mod

_3dgrt_plugin = None


def load_3dgrt_plugin(conf):
    global _3dgrt_plugin
    if _3dgrt_plugin is None:
        try:
            # Case 1: already importable from inside the package (e.g., if you ship the .so)
            from . import lib3dgrt_cc as tdgrt  # type: ignore
        except ImportError:
            # Case 2: build, then load from torch cache
            from .setup_3dgrt import setup_3dgrt
            maybe_mod = setup_3dgrt(conf)

            # If setup_3dgrt returns the module, use it; otherwise, pull it from cache.
            if maybe_mod is not None:
                tdgrt = maybe_mod
                # Make absolute and package-relative imports work later
                sys.modules['lib3dgrt_cc'] = tdgrt
                sys.modules['threedgrt_tracer.lib3dgrt_cc'] = tdgrt
            else:
                tdgrt = _load_from_torch_cache("lib3dgrt_cc")

        _3dgrt_plugin = tdgrt
    return _3dgrt_plugin

#def load_3dgrt_plugin(conf):
#    global _3dgrt_plugin
#    if _3dgrt_plugin is None:
#        try:
#            from . import lib3dgrt_cc as tdgrt  # type: ignore
#        except ImportError:
#            from .setup_3dgrt import setup_3dgrt
#
#            setup_3dgrt(conf)
#            import lib3dgrt_cc as tdgrt  # type: ignore
#        _3dgrt_plugin = tdgrt


# ----------------------------------------------------------------------------
#
class Tracer:
    class _Autograd(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            tracer_wrapper,
            frame_id,
            ray_to_world,
            ray_ori,
            ray_dir,
            mog_pos,
            mog_rot,
            mog_scl,
            mog_dns,
            mog_sph,
            render_opts,
            sph_degree,
            min_transmittance,
        ):
            particle_density = torch.concat([mog_pos, mog_dns, mog_rot, mog_scl, torch.zeros_like(mog_dns)], dim=1)
            ray_radiance, ray_density, ray_hit_distance, ray_normals, hits_count, mog_visibility = tracer_wrapper.trace(
                frame_id,
                ray_to_world,
                ray_ori,
                ray_dir,
                particle_density,
                mog_sph,
                render_opts,
                sph_degree,
                min_transmittance,
            )
            ctx.save_for_backward(
                ray_to_world,
                ray_ori,
                ray_dir,
                ray_radiance,
                ray_density,
                ray_hit_distance,
                ray_normals,
                particle_density,
                mog_sph,
            )
            ctx.frame_id = frame_id
            ctx.render_opts = render_opts
            ctx.sph_degree = sph_degree
            ctx.min_transmittance = min_transmittance
            ctx.tracer_wrapper = tracer_wrapper
            return (
                ray_radiance,
                ray_density,
                ray_hit_distance[:, :, :, 0:1],  # return only the hit distance
                ray_normals,
                hits_count,
                mog_visibility,
            )

        @staticmethod
        def backward(
            ctx,
            ray_radiance_grd,
            ray_density_grd,
            ray_hit_distance_grd,
            ray_normals_grd,
            ray_hits_count_grd_UNUSED,
            mog_visibility_grd_UNUSED,
        ):
            (
                ray_to_world,
                ray_ori,
                ray_dir,
                ray_radiance,
                ray_density,
                ray_hit_distance,
                ray_normals,
                particle_density,
                mog_sph,
            ) = ctx.saved_variables
            frame_id = ctx.frame_id
            particle_density_grd, mog_sph_grd = ctx.tracer_wrapper.trace_bwd(
                frame_id,
                ray_to_world,
                ray_ori,
                ray_dir,
                ray_radiance,
                ray_density,
                ray_hit_distance,
                ray_normals,
                particle_density,
                mog_sph,
                ray_radiance_grd,
                ray_density_grd,
                ray_hit_distance_grd,
                ray_normals_grd,
                ctx.render_opts,
                ctx.sph_degree,
                ctx.min_transmittance,
            )
            mog_pos_grd, mog_dns_grd, mog_rot_grd, mog_scl_grd, _ = torch.split(
                particle_density_grd, [3, 1, 4, 3, 1], dim=1
            )
            return (
                None,
                None,
                None,
                None,
                None,
                mog_pos_grd,
                mog_rot_grd,
                mog_scl_grd,
                mog_dns_grd,
                mog_sph_grd,
                None,
                None,
                None,
            )

    class RenderOpts(IntEnum):
        NONE = 0
        DEFAULT = NONE

    def __init__(self, conf):

        self.device = "cuda"
        self.conf = conf
        self.num_update_bvh = 0

        logger.info(f'🔆 Creating Optix tracing pipeline.. Using CUDA path: "{torch.utils.cpp_extension.CUDA_HOME}"')
        torch.zeros(1, device=self.device)  # Create a dummy tensor to force cuda context init
        load_3dgrt_plugin(conf)

        self.tracer_wrapper = _3dgrt_plugin.OptixTracer(
            os.path.dirname(__file__),
            torch.utils.cpp_extension.CUDA_HOME,
            self.conf.render.pipeline_type,
            self.conf.render.backward_pipeline_type,
            self.conf.render.primitive_type,
            self.conf.render.particle_kernel_degree,
            self.conf.render.particle_kernel_min_response,
            self.conf.render.particle_kernel_density_clamping,
            self.conf.render.particle_radiance_sph_degree,
            self.conf.render.enable_normals,
            self.conf.render.enable_hitcounts,
        )

        self.frame_timer = CudaTimer() if self.conf.render.enable_kernel_timings else None
        self.timings = {}

    def build_acc(self, gaussians, rebuild=True):
        with torch.cuda.nvtx.range(f"build-bvh-full-build-{rebuild}"):
            allow_bvh_update = (
                self.conf.render.max_consecutive_bvh_update > 1
            ) and not self.conf.render.particle_kernel_density_clamping
            rebuild_bvh = (
                rebuild
                or self.conf.render.particle_kernel_density_clamping
                or self.num_update_bvh >= self.conf.render.max_consecutive_bvh_update
            )
            self.tracer_wrapper.build_bvh(
                gaussians.positions.view(-1, 3).contiguous(),
                gaussians.rotation_activation(gaussians.rotation).view(-1, 4).contiguous(),
                gaussians.scale_activation(gaussians.scale).view(-1, 3).contiguous(),
                gaussians.density_activation(gaussians.density).view(-1, 1).contiguous(),
                rebuild_bvh,
                allow_bvh_update,
            )
            self.num_update_bvh = 0 if rebuild_bvh else self.num_update_bvh + 1

    def render(self, gaussians, gpu_batch: Batch, train=False, frame_id=0):
        num_gaussians = gaussians.num_gaussians
        with torch.cuda.nvtx.range(f"model.forward({num_gaussians} gaussians)"):

            if self.frame_timer is not None:
                self.frame_timer.start()

            (pred_rgb, pred_opacity, pred_dist, pred_normals, hits_count, mog_visibility) = Tracer._Autograd.apply(
                self.tracer_wrapper,
                frame_id,
                gpu_batch.T_to_world.contiguous(),
                gpu_batch.rays_ori.contiguous(),
                gpu_batch.rays_dir.contiguous(),
                gaussians.positions.contiguous(),
                gaussians.get_rotation().contiguous(),
                gaussians.get_scale().contiguous(),
                gaussians.get_density().contiguous(),
                gaussians.get_features().contiguous(),
                Tracer.RenderOpts.DEFAULT,
                gaussians.n_active_features,
                self.conf.render.min_transmittance,
            )

            if self.frame_timer is not None:
                self.frame_timer.end()

            pred_rgb, pred_opacity = gaussians.background(
                gpu_batch.T_to_world.contiguous(), gpu_batch.rays_dir.contiguous(), pred_rgb, pred_opacity, train
            )

        if self.frame_timer is not None:
            self.timings["forward_render"] = self.frame_timer.timing()

        return {
            "pred_rgb": pred_rgb,
            "pred_opacity": pred_opacity,
            "pred_dist": pred_dist,
            "pred_normals": torch.nn.functional.normalize(pred_normals, dim=3),
            "hits_count": hits_count,
            "frame_time_ms": self.frame_timer.timing() if self.frame_timer is not None else 0.0,
            "mog_visibility": mog_visibility,
        }
