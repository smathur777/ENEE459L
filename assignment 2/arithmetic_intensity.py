from __future__ import annotations

from math import prod
from typing import Any


FLOPS_PER_MAC = 2
BN_BUFFERS_PER_CHANNEL = 2
BUFFER_DTYPE = "fp32"


def computed(value: Any, message: str, **metadata: Any) -> dict[str, Any]:
    return {"type": "computed", "value": value, "message": message, **metadata}


def dtype_bytes(dtype: str) -> float:
    widths = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5}
    return widths[dtype]


def _layer_parameters(ly: Layer) -> int:
    if ly.kind == "bn":
        return 2 * ly.out_shape[0]
    if ly.kind == "conv":
        outputs = ly.out_shape[0]
        weights = outputs * (ly.in_shape[0] // ly.groups) * prod(ly.kernel or (1, 1))
    elif ly.kind == "linear":
        outputs = ly.out_elements
        weights = ly.in_elements * outputs
    else:
        return 0
    return weights + (outputs if ly.bias else 0)


def _layer_macs(ly: Layer) -> int:
    if ly.kind == "conv":
        return ly.out_elements * (ly.in_shape[0] // ly.groups) * prod(ly.kernel or (1, 1))
    if ly.kind == "linear":
        return ly.in_elements * ly.out_elements
    return 0


def arithmetic_intensity(graph: Graph) -> dict[str, Any]:
    layers = list(graph)
    by_name = {ly.name: ly for ly in layers}
    per_layer: dict[str, dict[str, Any]] = {}
    total_flops = 0
    total_bytes = 0.0

    for i, ly in enumerate(layers):
        if ly.reads:
            in_dtype = by_name[ly.reads[0]].act_dtype
        elif i > 0:
            in_dtype = layers[i - 1].act_dtype
        else:
            in_dtype = graph.precision

        in_b = ly.in_elements * dtype_bytes(in_dtype)
        if ly.kind == "add" and len(ly.reads) == 2:
            in_b *= 2

        weight_b = _layer_parameters(ly) * dtype_bytes(ly.weight_dtype)
        if ly.kind == "bn":
            weight_b += (
                BN_BUFFERS_PER_CHANNEL
                * ly.out_shape[0]
                * dtype_bytes(BUFFER_DTYPE)
            )

        out_b = ly.out_elements * dtype_bytes(ly.act_dtype)
        moved_b = float(in_b + weight_b + out_b)
        flops = _layer_macs(ly) * FLOPS_PER_MAC

        per_layer[ly.name] = {
            "flops": flops,
            "bytes": moved_b,
            "intensity": round(flops / moved_b, 4) if moved_b else 0.0,
        }
        total_flops += flops
        total_bytes += moved_b

    return computed(
        round(total_flops / total_bytes, 4) if total_bytes else 0.0,
        f"{graph.name}: FLOPs over bytes, every byte counted once",
        per_layer=per_layer,
        total_flops=total_flops,
        total_bytes=total_bytes,
        flops_per_mac=FLOPS_PER_MAC,
        cache_reuse_assumed=False,
        note="no fusion and no cache reuse; both would lower the byte column",
    )
