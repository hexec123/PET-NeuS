"""
Multi‑GPU training runner for PET‑NeuS
--------------------------------------

This module introduces a `MultiGPURunner` class which extends the
standard `Runner` from ``exp_runner_pet.py`` to support training
across multiple GPUs.  The original PET‑NeuS implementation
assumed a single GPU selected via ``--gpu`` and set the default
CUDA device accordingly.  On systems with multiple GPUs (e.g. a
dual RTX 5090 workstation), wrapping the networks in
``torch.nn.DataParallel`` allows PyTorch to distribute the
forward/backward computation over several devices, potentially
reducing training time.  This wrapper does not modify the core
algorithm or dataset logic of PET‑NeuS; instead it leverages
PyTorch’s built‑in data parallelism.

Usage
-----
```
python exp_runner_pet_multi.py \
    --conf ./confs/womask_pet.conf \
    --mode train \
    --case my_scene \
    --gpu_ids 0,1 \
    --base_exp_dir ./my_experiment
```

The ``--gpu_ids`` flag accepts a comma‑separated list of GPU
indices.  The first ID in the list is treated as the primary
device for tensor allocation and logging; additional devices are
used by ``DataParallel``.  When only a single GPU is specified,
the behaviour reduces to the original single‑GPU code.

Notes
-----
* Installing the CUDA‑enabled PyTorch wheel for your hardware
  (e.g. CUDA 12.6) is required to make full use of multiple GPUs.
  See the updated ``requirements.txt`` for details【855632448272888†L139-L176】.
* Data parallelism replicates the neural networks on each GPU
  and splits input batches automatically.  For very large scenes,
  consider reducing the batch size to avoid running out of memory.
* This script lives in a separate file so it can reside in a
  dedicated ``dual‑gpu`` branch without interfering with the
  original ``exp_runner_pet.py``.
"""

import argparse
import logging
import os
import torch

from exp_runner_pet import Runner as BaseRunner
from models.renderer_pet import NeuSRenderer


class MultiGPURunner(BaseRunner):
    """Extension of the PET‑NeuS Runner with multi‑GPU support.

    Parameters
    ----------
    conf_path : str
        Path to the configuration (.conf) file.
    mode : str, optional
        Training or evaluation mode, by default ``'train'``.
    case : str, optional
        Name of the dataset case, used to replace ``CASE_NAME``
        in the configuration, by default ``'CASE_NAME'``.
    is_continue : bool, optional
        Whether to resume training from a checkpoint, by default False.
    ckpt_name : str or None, optional
        Name of the checkpoint to load when resuming, by default None.
    base_exp_dir : str or None, optional
        Directory to store experiments; overrides the value in the
        configuration when provided.
    end_iter : int or None, optional
        Maximum training iterations; overrides the configuration when
        provided.
    gpu_ids : str or sequence[int], optional
        Comma‑separated list of GPU indices (e.g. ``'0,1'``).  When
        more than one ID is supplied, networks are wrapped in
        :class:`torch.nn.DataParallel`.
    """

    def __init__(
        self,
        conf_path: str,
        mode: str = 'train',
        case: str = 'CASE_NAME',
        is_continue: bool = False,
        ckpt_name: str | None = None,
        base_exp_dir: str | None = None,
        end_iter: int | None = None,
        gpu_ids: str | list[int] = '0',
    ) -> None:
        # Parse the GPU IDs into a list of ints
        if isinstance(gpu_ids, str):
            self.gpu_ids = [int(x) for x in gpu_ids.split(',') if x.strip() != '']
        else:
            self.gpu_ids = list(gpu_ids)
        # Always set the primary device to the first GPU in the list
        primary_gpu = self.gpu_ids[0] if self.gpu_ids else 0
        torch.cuda.set_device(primary_gpu)
        self.device = torch.device(f'cuda:{primary_gpu}')

        # Initialise the base Runner (this builds the networks and loads the dataset)
        super().__init__(conf_path, mode, case, is_continue, ckpt_name, base_exp_dir, end_iter)

        # When using multiple GPUs, wrap networks in DataParallel.  We
        # re‑instantiate the renderer with the parallel networks so that
        # rendering calls route through the DataParallel wrappers.
        if len(self.gpu_ids) > 1:
            logging.info(f'Using DataParallel on devices {self.gpu_ids}')
            self.nerf_outside = torch.nn.DataParallel(self.nerf_outside, device_ids=self.gpu_ids)
            self.sdf_network = torch.nn.DataParallel(self.sdf_network, device_ids=self.gpu_ids)
            self.deviation_network = torch.nn.DataParallel(self.deviation_network, device_ids=self.gpu_ids)
            self.color_network = torch.nn.DataParallel(self.color_network, device_ids=self.gpu_ids)
            # Wrap LPIPS network as well to avoid device mismatches
            self.lpips_vgg_fn = torch.nn.DataParallel(self.lpips_vgg_fn, device_ids=self.gpu_ids)
            # Rebuild the renderer using the parallel networks
            self.renderer = NeuSRenderer(
                self.nerf_outside,
                self.sdf_network,
                self.deviation_network,
                self.color_network,
                **self.conf['model.neus_renderer'],
            )
            # Ensure training images reside on the primary device
            self.train_images = self.train_images.to(self.device)


def main() -> None:
    """Entry point for the multi‑GPU PET‑NeuS runner."""
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser(description='PET‑NeuS multi‑GPU runner')
    parser.add_argument('--conf', type=str, default='./confs/base.conf', help='path to configuration file')
    parser.add_argument('--mode', type=str, default='train', help='mode: train | validate_mesh | validate_image | interpolate_i_j')
    parser.add_argument('--mcube_threshold', type=float, default=0.0, help='threshold for marching cubes')
    parser.add_argument('--is_continue', default=False, action='store_true', help='continue from latest checkpoint')
    parser.add_argument('--ckpt_name', type=str, default=None, help='name of the checkpoint to load')
    parser.add_argument('--gpu_ids', type=str, default='0', help='comma‑separated list of GPU indices')
    parser.add_argument('--case', type=str, default='', help='case name for dataset')
    parser.add_argument('--image_idx', type=int, default=0, help='index of image for validation')
    parser.add_argument('--image_resolution', type=int, default=4, help='resolution level for validating images')
    parser.add_argument('--mesh_resolution', type=int, default=512, help='resolution of the marching cubes mesh')
    parser.add_argument('--base_exp_dir', type=str, default=None, help='directory to store experiments')
    parser.add_argument('--end_iter', type=int, default=None, help='maximum iteration for training')

    args = parser.parse_args()

    runner = MultiGPURunner(
        args.conf,
        args.mode,
        args.case,
        args.is_continue,
        args.ckpt_name,
        args.base_exp_dir,
        args.end_iter,
        gpu_ids=args.gpu_ids,
    )

    # Dispatch on mode (same as original exp_runner_pet)
    if args.mode == 'train':
        runner.train()
        runner.validate_mesh(world_space=True, resolution=args.mesh_resolution, threshold=args.mcube_threshold)
        runner.validate_image(idx=args.image_idx, resolution_level=args.image_resolution)
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=args.mesh_resolution, threshold=args.mcube_threshold)
    elif args.mode == 'validate_image':
        runner.validate_image(idx=args.image_idx, resolution_level=args.image_resolution)
    elif args.mode.startswith('interpolate'):
        # mode should be in the form interpolate_i_j
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
    else:
        raise ValueError(f'Unknown mode: {args.mode}')


if __name__ == '__main__':
    main()
