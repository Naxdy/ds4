# NVIDIA CUDA GPUs

[README](../README.md) | [Getting started](../README.md#start-here)

**Ada Lovelace is supported, including 48 GB L40S cards.** This is an important
use case: a model whose other inference implementations require newer GPUs
can still run here when DwarfStar supports its GGUF layout. The Flash kernels
do not require Blackwell's native FP4 instructions; Ada uses an appropriate
CUDA path. Support remains model-specific, not a promise to run arbitrary GGUFs.

## Build

Install the NVIDIA driver and CUDA toolkit, including `nvcc` and cuBLAS.
Use the local GPU architecture, or select Ada explicitly:

```sh
make cuda-generic
# For an Ada build:
make cuda CUDA_ARCH=sm_89
```

For a GB10 machine use the [DGX Spark](DGX_SPARK.md) target instead.

## Normal layer placement

Without tensor parallelism, DwarfStar places complete layers on the selected
GPUs. `--gpu-vram auto` uses reported free VRAM, reserving space for the graph
and context. Explicit budgets are comma-separated GiB values, one per device.

```sh
./download_model.sh ds4f-q2
./ds4 --cuda --gpu-devices 0,1,2,3 --gpu-vram auto --ctx 32768
```

This is also the multi-GPU placement mode for GLM. Startup refuses a layout
that would require unsupported CPU execution; reduce context or choose a
smaller model if necessary.

## Flash tensor parallelism

`--cuda-tensor-parallel` pairs GPUs and divides routed-expert work inside
each pair. Pairs also own successive layer ranges. This is an in-process
configuration, not the network `--role coordinator` / `--role worker` mode.

The device order matters. List all layer-home GPUs first, then their partners.
For physical pairs `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`, use:

```text
0,2,4,6,1,3,5,7
```

Check your machine's topology with `nvidia-smi topo -m`; do not assume this
ordering gives the best pairs on another server. Each pair stores one half
of the routed experts per GPU. Dense attention, routers, and shared experts
are replicated within the pair; the output head is vocabulary-sharded.

For the tested eight-L40S setup:

```sh
./download_model.sh ds4f-q4
./ds4-agent --cuda --cuda-tensor-parallel \
  --gpu-devices 0,2,4,6,1,3,5,7 --gpu-vram auto --ctx 100000
```

Q4 has native grouped routed kernels for multi-user throughput. Q2 needs less
memory, but unsupported grouped shapes use a slower, correct fallback.
Four 48 GB cards with Q2 and eight with Q4 are tested configurations. An even
device count alone does not guarantee memory fit.

## Serve multiple users

```sh
./ds4-server --cuda --cuda-tensor-parallel \
  --gpu-devices 0,2,4,6,1,3,5,7 --gpu-vram auto \
  --ctx 100000 --batched-session 16 --host 0.0.0.0
```

The supported eight-L40S Flash configuration has reached roughly 126 aggregate
generation tokens/s with 16 decode rows. This is total throughput, not the
speed of one user. The recorded conditions and regression requirements are in
[the QA guide](../QA_BEFORE_RELEASES.md).

CUDA TP defaults to 2048-token prefill chunks. Keep that default for the
16-session, 100k-context setup; larger chunks need more scratch memory.
Reduce session count or context if all KV states do not fit.

The repository's [server launcher](../run-nvidia-tp-server.sh) is an example
from the L40S deployment, not a portable default: it uses host-specific model
and cache paths and defaults to MXFP4. The commands above need no launcher or
environment tuning. See [serving](SERVER.md) for disk caches and API access.

## DeepSeek V4.1 Flash: in-process tensor parallelism (Blackwell)

`--cuda-tensor-parallel` also works for **DeepSeek V4.1 Flash** on a single
two-GPU host (e.g. a Blackwell DB/GB pair). Unlike the Flash (DeepSeek V4)
placement mode above, V4.1 keeps each rank on **exactly one GPU** in a
genuinely single-process configuration: no child process, no exec, no
loopback network, no coordinator role. The leader frontend (ds4, ds4-server,
ds4-agent, ds4-bench) initializes both CUDA devices in this one process, runs
rank 0 on the first device, and mirrors rank 1 as a thread of the same
process on the second device. Every gate exchange and lockstep command
travels over an in-process socketpair; the proven two-rank protocol is
reused verbatim, so the two ranks are behaviorally identical to a
machine-to-machine `--tensor-parallel` pair.

```sh
./ds4-agent --cuda --cuda-tensor-parallel \
  --gpu-devices 0,1 --gpu-vram auto --ctx 100000

./ds4-server --cuda --cuda-tensor-parallel \
  --gpu-devices 1,0 --gpu-vram auto --ctx 100000 \
  --batched-session 16 --host 0.0.0.0
```

Requirements and semantics match the QA'd network V4.1 TP path:

- Exactly **two** CUDA devices are accepted (`--gpu-devices` is optional; an
  explicit `--gpu-vram` budget for device 1 is honored, otherwise each rank
  auto-detects its own free VRAM on its card). The first listed device is
  rank 0, the second is rank 1. Device order does not imply pairing layers —
  both ranks run the complete layer stack on their own GPU.
- The routed experts must be IQ2_XXS gate/up and Q2_K down (the V4.1 Q2
  recipe). Dense/shared weights replicate on both ranks; each rank's weights
  live only on its own device, and the output head is vocabulary-split.
- Quality mode (`--quality`), SSD streaming, DSpark and steering remain
  unsupported under TP, exactly as in the network configuration.
- Both ranks' logs appear on the same terminal (it is one process); shutdown
  closes the transport and joins the worker thread, so no orphaned process
  or zombie can be left behind.

The 2048-token prefill chunk note above does not apply: V4.1 prefill is
driven by the leader and chunked identically on both ranks, so the leader
and worker never desync over differing chunk settings.

