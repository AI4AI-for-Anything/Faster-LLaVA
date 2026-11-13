"""Shared setup for torch_xla's Pallas splash attention (used by hf_splash.py).

Holds the mesh that splash runs under (it is called inside shard_map and needs
the mesh name and how q/k/v are sharded), builds the SplashAttentionConfig, and
pins the Mosaic IR version the installed libtpu can read.

Requires the torch_xla pallas extra (jax 0.7.1).
"""
from torch_xla.experimental.splash_attention import SplashAttentionConfig


def _pin_mosaic_ir_version(version=7):
    """Emit Mosaic IR the installed libtpu can read.

    torch_xla's pallas extra pins jax 0.7.1, which serialises Mosaic kernels at
    IR version 8 unless it detects an older Cloud TPU runtime. Under call_jax it
    does not (jax has no TPU backend of its own here), so libtpu 0.0.21 rejected
    the kernel: "Failed to deserialize the Mosaic module: Unsupported version:
    expected <= 7 but got 8". jax's own rule maps runtimes built before
    2025-08-14 to version 7; this applies that rule explicitly instead of
    editing site-packages.
    """
    from jax._src import tpu_custom_call as tcc
    tcc.get_ir_version = lambda ctx: version


_pin_mosaic_ir_version()


def _config(T, window, mesh_name, qkv_spec):
    blk = min(512, T)
    return SplashAttentionConfig(
        sa_block_q=blk, sa_block_kv=blk, sa_block_kv_compute=blk,
        sa_block_q_dkv=blk, sa_block_kv_dkv=blk, sa_block_kv_dkv_compute=blk,
        sa_block_q_dq=blk, sa_block_kv_dq=blk,
        mesh=mesh_name, qkv_partition_spec=qkv_spec,
        segment_ids_partition_spec=(qkv_spec[0], None),
        attentiontype_local_sliding=window is not None,
        slide_window_size=window).to_json()


_MESH = {"name": None, "spec": (("data",), None, None, None)}


def set_mesh(mesh, batch_axes):
    """Splash runs under shard_map; it needs the mesh and how q/k/v are sharded."""
    _MESH["name"] = str(mesh)
    ax = batch_axes if isinstance(batch_axes, tuple) else (batch_axes,)
    _MESH["spec"] = (ax, None, None, None)
