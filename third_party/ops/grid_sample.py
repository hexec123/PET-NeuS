import torch
from torch.utils.cpp_extension import load as _cpp_load
from pkg_resources import parse_version
import os
import warnings

gridsample_grad2 = None


def _build_if_needed():
    """Build the CUDA grad-grad op from .cpp+.cu sources, if available."""
    global gridsample_grad2
    if gridsample_grad2 is not None:
        return True

    srcdir = os.path.dirname(__file__)
    sources = [
        os.path.join(srcdir, "gridsample_bindings.cpp"),
        os.path.join(srcdir, "gridsample_cuda.cu"),
    ]
    try:
        gridsample_grad2 = _cpp_load(
            name="gridsample_grad2",
            sources=sources,
            extra_cflags=[],
            extra_cuda_cflags=["--use_fast_math"],
            verbose=True,
        )
        return True
    except Exception as e:
        warnings.warn(
            f"[grid_sample] Falling back to PyTorch autograd (no grad-grad): {e}"
        )
        gridsample_grad2 = None
        return False


# Build (or confirm) once on import
_build_if_needed()

_use_pytorch_1_11_api = parse_version(torch.__version__) >= parse_version("1.11.0a")


def grid_sample_2d(input, grid, padding_mode="zeros", align_corners=True):
    assert padding_mode in ["zeros", "border"]
    return _GridSample2dForward.apply(input, grid, padding_mode, align_corners)


def grid_sample_3d(input, grid, padding_mode="zeros", align_corners=True):
    assert padding_mode in ["zeros", "border"]
    return _GridSample3dForward.apply(input, grid, padding_mode, align_corners)


class _GridSample2dForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, padding_mode="zeros", align_corners=True):
        assert (
            input.ndim == 4
            and grid.ndim == 4
            and input.shape[0] == grid.shape[0]
            and grid.shape[-1] == 2
        )
        out = torch.nn.functional.grid_sample(
            input=input,
            grid=grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        ctx.save_for_backward(input, grid)
        ctx.padding_mode = ["zeros", "border"].index(padding_mode)
        ctx.align_corners = align_corners
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = _GridSample2dBackward.apply(
            grad_output, input, grid, ctx.padding_mode, ctx.align_corners
        )
        return grad_input, grad_grid, None, None


class _GridSample2dBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid, padding_mode=0, align_corners=True):
        op = torch._C._jit_get_operation("aten::grid_sampler_2d_backward")[0]
        if _use_pytorch_1_11_api:
            output_mask = (True, True)
            grad_input, grad_grid = op(
                grad_output, input, grid, 0, padding_mode, align_corners, output_mask
            )
        else:
            grad_input, grad_grid = op(
                grad_output, input, grid, 0, padding_mode, align_corners
            )
        ctx.save_for_backward(grad_output, input, grid)
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        if gridsample_grad2 is None:
            grad_output, input, grid = ctx.saved_tensors
            zero_like = lambda t: torch.zeros_like(t) if t is not None else None
            return (
                zero_like(grad_output),
                zero_like(input),
                zero_like(grid),
                None,
                None,
            )

        grad_output, input, grid = ctx.saved_tensors
        grad_grad_output, grad_input, grad_grid = gridsample_grad2.grad2_2d(
            grad2_grad_input,
            grad2_grad_grid,
            grad_output,
            input,
            grid,
            ctx.padding_mode,
            ctx.align_corners,
        )
        return grad_grad_output, grad_input, grad_grid, None, None


class _GridSample3dForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, padding_mode="zeros", align_corners=True):
        assert (
            input.ndim == 5
            and grid.ndim == 5
            and input.shape[0] == grid.shape[0]
            and grid.shape[-1] == 3
        )
        out = torch.nn.functional.grid_sample(
            input=input,
            grid=grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        ctx.save_for_backward(input, grid)
        ctx.padding_mode = ["zeros", "border"].index(padding_mode)
        ctx.align_corners = align_corners
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = _GridSample3dBackward.apply(
            grad_output, input, grid, ctx.padding_mode, ctx.align_corners
        )
        return grad_input, grad_grid, None, None


class _GridSample3dBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid, padding_mode=0, align_corners=True):
        op = torch._C._jit_get_operation("aten::grid_sampler_3d_backward")[0]
        if _use_pytorch_1_11_api:
            output_mask = (True, True)
            grad_input, grad_grid = op(
                grad_output, input, grid, 0, padding_mode, align_corners, output_mask
            )
        else:
            grad_input, grad_grid = op(
                grad_output, input, grid, 0, padding_mode, align_corners
            )
        ctx.save_for_backward(grad_output, input, grid)
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        if gridsample_grad2 is None:
            grad_output, input, grid = ctx.saved_tensors
            zero_like = lambda t: torch.zeros_like(t) if t is not None else None
            return (
                zero_like(grad_output),
                zero_like(input),
                zero_like(grid),
                None,
                None,
            )

        grad_output, input, grid = ctx.saved_tensors
        grad_grad_output, grad_input, grad_grid = gridsample_grad2.grad2_3d(
            grad2_grad_input,
            grad2_grad_grid,
            grad_output,
            input,
            grid,
            ctx.padding_mode,
            ctx.align_corners,
        )
        return grad_grad_output, grad_input, grad_grid, None, None
