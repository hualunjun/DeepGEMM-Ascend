"""Golden reference for MegaMoE W4A4 Linear1 + W4A8 Linear2.

The golden mirrors the kernel's arithmetic:
  - FP4 E2M1 unpacking with low nibble = even K, high nibble = odd K.
  - UE8M0 scale bytes: scale = 2 ** (exponent - 127), one scale per 32 K elements.
  - Native Linear1 FP4 x FP4 with FP32 accumulation.
  - SwiGLU clamp + FP8 E4M3 quantization using the same power-of-two scale
    bit trick as vf_swiglu_quant.
  - Linear2 FP8 x FP4 (dequantized to its true E2M1 value) with FP32 accumulation.
  - BF16 rounding of each route output before the FP32 combine sum.

Run (single rank, all experts local):
  PYTHONPATH=/path/to/tilelang_stub:$PWD/tests:$PWD \
  python3 tests/golden_mega_moe.py

The script first checks that the NPU is idle; if another process is using it,
the script exits without touching the device.
"""

import argparse
import os
import shutil
import subprocess
import sys

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from utils import calc_diff
from test_mega_moe import MegaMoECase, make_npu_fp4
from deep_gemm.testing.envs import init_dist


# E2M1 values indexed by the 4-bit code.
_FP4_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _npu_is_idle(device_idx: int = 0) -> bool:
    exe = shutil.which("npu-smi") or "/usr/local/sbin/npu-smi"
    try:
        out = subprocess.run([exe, "info"], capture_output=True, text=True, timeout=20).stdout
    except Exception as exc:
        print(f"[warn] cannot query npu-smi ({exc}); assuming idle")
        return True
    marker = f"No running processes found in NPU {device_idx}"
    return marker in out


def decode_fp4(data: torch.Tensor, sf: torch.Tensor, num_cols: int) -> torch.Tensor:
    """Decode packed FP4 + UE8M0 SF to FP32 values.

    data: [..., num_cols // 2] int8
    sf:   [..., num_cols // 64] int16 (two UE8M0 bytes per int16)
    """
    packed = data.view(torch.uint8)
    low = (packed & 0x0F).long()
    high = (packed >> 4).long()
    lut = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=data.device)
    values = torch.empty(*packed.shape[:-1], num_cols, dtype=torch.float32, device=data.device)
    values[..., 0::2] = lut[low]
    values[..., 1::2] = lut[high]

    # int16 -> two little-endian exponent bytes, one per 32 K elements.
    sf_bytes = sf.view(torch.uint8)
    scale = torch.pow(torch.tensor(2.0, device=data.device), sf_bytes.float() - 127.0)
    scale = scale.repeat_interleave(32, dim=-1)[..., :num_cols]
    return values * scale


def quantize_fp8_golden(values: torch.Tensor, return_scale: bool = False):
    """Match vf_swiglu_quant: per-32 UE8M0 scale, E4M3 RN sat, return dequant values."""
    num_cols = values.shape[-1]
    assert num_cols % 32 == 0
    groups = values.reshape(*values.shape[:-1], num_cols // 32, 32)

    amax = groups.abs().amax(dim=-1).clamp_min(1e-4)
    bits = amax.view(torch.int32)
    bits = torch.maximum(bits, torch.full_like(bits, 0x38D1B717))  # 1e-4
    bits = bits + 0x001FFFFF
    bits = bits & 0x7F800000
    bits = bits - 0x04000000
    bits = torch.minimum(bits, torch.full_like(bits, 0x7F000000))
    scale = bits.view(torch.float32)
    inv_scale = (0x7F000000 - bits).view(torch.float32)

    scaled = groups * inv_scale[..., None]
    quant = scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    dequant = quant.float() * scale[..., None]
    dequant = dequant.reshape(*values.shape[:-1], num_cols)
    if return_scale:
        return dequant, scale
    return dequant


def swiglu_quant(
    gate: torch.Tensor,
    up: torch.Tensor,
    route_weights: torch.Tensor | None,
    activation_clamp: float,
) -> torch.Tensor:
    """Match vf_swiglu_quant: clamp, SwiGLU, route weight, FP8 quant, return dequant values."""
    gate = torch.minimum(gate, torch.tensor(activation_clamp, device=gate.device))
    up = torch.clamp(up, -activation_clamp, activation_clamp)
    # Kernel computes gate / (1 + exp(-gate)) * up.
    value = gate * torch.sigmoid(gate) * up
    if route_weights is not None:
        value = value * route_weights[:, None]
    return quantize_fp8_golden(value)

def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int32)
    exp = ((bits // (2 ** 23)) % 256) + (bits % (2 ** 23)).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def _rne_e2m1(x: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(_FP4_VALUES[:8], dtype=torch.float32, device=x.device)
    sign = torch.sign(x)
    ax = x.abs().clamp(max=6.0)
    d = (ax.unsqueeze(-1) - table).abs()
    lo = d.argmin(-1)
    lo_val = table[lo]
    hi = (lo + 1).clamp(max=7)
    hi_val = table[hi]
    d_lo, d_hi = (ax - lo_val).abs(), (hi_val - ax).abs()
    tie = (d_lo == d_hi) & (lo < 7)
    idx = torch.where(tie & (lo % 2 == 1), hi, lo)
    return table[idx] * torch.where(sign < 0, -1.0, 1.0)


def _kernel_quant_e2m1(a: torch.Tensor) -> torch.Tensor:
    shape = a.shape
    groups = a.reshape(*shape[:-1], shape[-1] // 32, 32)
    sf = _ceil_to_ue8m0(groups.abs().amax(dim=-1).clamp(1e-4) / 6.0)
    sfull = sf.repeat_interleave(32, dim=-1)
    scaled_bf16 = (a / sfull).to(torch.bfloat16).float()
    return (_rne_e2m1(scaled_bf16) * sfull).reshape(shape)


def swiglu_quant_fp4(
    gate: torch.Tensor,
    up: torch.Tensor,
    route_weights: torch.Tensor | None,
    activation_clamp: float,
) -> torch.Tensor:
    """Match the spike A4W4 path: FP8 code / 64 -> E2M1, SF folded by 2^-6."""
    gate = torch.minimum(gate, torch.tensor(activation_clamp, device=gate.device))
    up = torch.clamp(up, -activation_clamp, activation_clamp)
    value = gate * torch.sigmoid(gate) * up
    if route_weights is not None:
        value = value * route_weights[:, None]
    return _kernel_quant_e2m1(value)


def golden_mega_moe(case: MegaMoECase, rank_idx: int) -> torch.Tensor:
    buffer = case.buffer
    num_tokens = case.y.size(0)
    hidden = buffer.hidden
    intermediate = buffer.intermediate_hidden
    num_experts = buffer.num_experts
    num_topk = buffer.num_topk
    shared = buffer.num_shared_experts
    clamp = case.activation_clamp

    # Recreate the raw weights with the same seeds as MegaMoECase.
    l1_data, l1_sf = make_npu_fp4((num_experts, 2 * intermediate, hidden), 20000 + rank_idx)
    l2_data, l2_sf = make_npu_fp4((num_experts, hidden, intermediate), 21000 + rank_idx)
    w1 = decode_fp4(l1_data, l1_sf, hidden)          # [E, 2I, H]
    w2 = decode_fp4(l2_data, l2_sf, intermediate)    # [E, H, I]

    shared_intermediate = shared * intermediate
    if shared:
        sl1_data, sl1_sf = make_npu_fp4((2 * shared_intermediate, hidden), 24000)
        sl2_data, sl2_sf = make_npu_fp4((hidden, shared_intermediate), 25000)
        sw1 = decode_fp4(sl1_data, sl1_sf, hidden)          # [2Is, H]
        sw2 = decode_fp4(sl2_data, sl2_sf, shared_intermediate)  # [H, Is]

    x = decode_fp4(case.x[0], case.x[1], hidden)     # [T, H]
    topk_idx = case.topk_idx
    topk_weights = case.topk_weights.float()

    out = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=case.y.device)

    # Group routes by local expert; all experts are local in the single-rank case.
    for expert in range(num_experts):
        route_mask = topk_idx == expert
        if not route_mask.any():
            continue
        token_idx, slot_idx = route_mask.nonzero(as_tuple=True)
        x_e = x[token_idx]                                  # [n, H]
        l1 = x_e @ w1[expert].T                             # [n, 2I]
        gate, up = l1[:, :intermediate], l1[:, intermediate:]
        if case.buffer.linear2_fp4:
            l2_act = swiglu_quant_fp4(gate, up, topk_weights[token_idx, slot_idx], clamp)
        else:
            l2_act = swiglu_quant(gate, up, topk_weights[token_idx, slot_idx], clamp)
        l2 = l2_act @ w2[expert].T                          # [n, H]
        # Kernel writes each route to BF16 before the combine accumulation.
        out.index_add_(0, token_idx, l2.to(torch.bfloat16).float())

    if shared:
        l1 = x @ sw1.T                                      # [T, 2Is]
        gate, up = l1[:, :shared_intermediate], l1[:, shared_intermediate:]
        if case.buffer.linear2_fp4:
            l2_act = swiglu_quant_fp4(gate, up, None, clamp)
        else:
            l2_act = swiglu_quant(gate, up, None, clamp)
        l2 = l2_act @ sw2.T                                 # [T, H]
        out += l2.to(torch.bfloat16).float()

    return out.to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser(description="MegaMoE golden reference checker")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate-hidden", type=int, default=2304)
    parser.add_argument("--num-experts", type=int, default=48)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--num-shared-experts", type=int, default=1)
    parser.add_argument("--activation-clamp", type=float, default=7.0)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--mma-type", default="fp4xfp4+fp8xfp4",
                        choices=("fp4xfp4+fp8xfp4", "fp8xfp4+fp8xfp4", "fp4xfp4+fp4xfp4"))
    args = parser.parse_args()

    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '0').split(',')[0]
    if not _npu_is_idle(int(visible)):
        print("[PAUSE] NPU 0 is busy; not starting the golden check.")
        return

    _, _, group = init_dist(0, 1)
    case = MegaMoECase(
        group,
        args.hidden,
        args.intermediate_hidden,
        args.num_experts,
        args.num_topk,
        args.num_tokens,
        num_shared_experts=args.num_shared_experts,
        activation_clamp=args.activation_clamp,
        with_baseline=False,
        mma_type=args.mma_type,
    )
    try:
        case.create_inputs(args.num_tokens)
        fused = case.run_fused().clone()
        torch.npu.synchronize()
        golden = golden_mega_moe(case, rank_idx=group.rank())

        diff = calc_diff(fused, golden)
        max_abs = (fused.float() - golden.float()).abs().max().item()
        print(f"[golden] tokens={args.num_tokens} diff={diff:.3e} max_abs={max_abs:.3e}")
        assert diff < args.threshold, f"golden diff too large: {diff:.3e}"
        print("[golden] PASS")
    finally:
        case.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
