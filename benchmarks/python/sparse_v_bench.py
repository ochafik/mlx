#!/usr/bin/env python3
"""
Benchmark: Sparse-V optimization modes in LUT SDPA kernel.

Compares three sparse-V strategies:
  Mode 0: Dense (no sparse-V, threshold ignored)
  Mode 1: Branch-based (per-position skip via SIMD-uniform branch)
  Mode 2: Compact-then-compute (batch K-scores, compact indices, dense V)

Tests with both synthetic data (controlled sparsity) and real model KV cache.

Usage:
  python sparse_v_bench.py                     # Synthetic benchmarks
  python sparse_v_bench.py --real-model        # Real model benchmarks
  python sparse_v_bench.py --real-model --model mlx-community/Qwen3.5-4B-MLX-4bit
"""

import argparse
import math
import time

import mlx.core as mx


# Lloyd-Max 4-bit centroids (standard normal, std=1)
CENTROIDS_4BIT = [
    -2.7331, -2.0698, -1.6189, -1.2570, -0.9431, -0.6573, -0.3884, -0.1285,
     0.1285,  0.3884,  0.6573,  0.9431,  1.2570,  1.6189,  2.0698,  2.7331,
]
BOUNDARIES_4BIT = [
    -2.4015, -1.8444, -1.4380, -1.1001, -0.8002, -0.5229, -0.2585,
     0.0000,
     0.2585,  0.5229,  0.8002,  1.1001,  1.4380,  1.8444,  2.4015,
]


def make_centroids(D, dtype=mx.float32):
    s = 1.0 / math.sqrt(D)
    return mx.array([c * s for c in CENTROIDS_4BIT], dtype=dtype)


def turboquant_pack_4bit(vectors: mx.array):
    """Pack vectors to 4-bit LUT format with norm separation."""
    B, H, N, D = vectors.shape
    assert D % 8 == 0
    norms = mx.linalg.norm(vectors, axis=-1)
    unit = vectors / mx.maximum(norms[..., None], mx.array(1e-8))
    scaled = (unit * math.sqrt(D)).astype(mx.float32)
    boundaries = mx.array(BOUNDARIES_4BIT, dtype=mx.float32)
    indices = mx.zeros(scaled.shape, dtype=mx.uint32)
    for bv in boundaries:
        indices = indices + (scaled > bv).astype(mx.uint32)
    indices = indices.reshape(B, H, N, D // 8, 8)
    packed = mx.zeros((B, H, N, D // 8), dtype=mx.uint32)
    for i in range(8):
        packed = packed | (indices[..., i] << (i * 4))
    return packed, norms


def cossim(a: mx.array, b: mx.array) -> float:
    a_f = a.reshape(-1).astype(mx.float32)
    b_f = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_f * b_f)
    na = mx.linalg.norm(a_f)
    nb = mx.linalg.norm(b_f)
    return (dot / mx.maximum(na * nb, mx.array(1e-10))).item()


def benchmark_fn(fn, warmup=10, iters=100):
    """Benchmark a function, returning median latency in ms."""
    for _ in range(warmup):
        mx.eval(fn())
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def compute_attention_sparsity(q, k, scale, threshold):
    """Fraction of softmax weights below threshold."""
    B, Hq, _, D = q.shape
    Hkv = k.shape[1]
    gs = Hq // Hkv
    parts = []
    for g in range(Hkv):
        qg = q[:, g*gs:(g+1)*gs, :, :]
        kg = k[:, g:g+1, :, :]
        parts.append((qg @ kg.transpose(0, 1, 3, 2)) * scale)
    scores = mx.concatenate(parts, axis=1)
    attn = mx.softmax(scores, axis=-1)
    return (attn < threshold).astype(mx.float32).mean().item()


MODE_NAMES = {0: "dense", 1: "branch", 2: "compact"}


def run_synthetic_bench(args):
    """Synthetic benchmark with controlled parameters."""
    print("=" * 80)
    print("SYNTHETIC SPARSE-V BENCHMARK")
    print("=" * 80)
    print()

    D = args.head_dim
    bits = 4
    pf = 32 // bits
    B = 1
    H = args.n_heads
    scale = 1.0 / math.sqrt(D)
    dtype = mx.float32 if D == 256 else mx.bfloat16

    ck = make_centroids(D, dtype=dtype)
    cv = make_centroids(D, dtype=dtype)

    thresholds = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2]

    for N in args.seq_lengths:
        print(f"\n{'_' * 78}")
        print(f"D={D}  N={N}  H={H}  bits={bits}  dtype={dtype}")
        print(f"{'_' * 78}")

        q = mx.random.normal((B, H, 1, D)).astype(dtype)
        k_data = mx.random.normal((B, H, N, D)).astype(dtype)
        v_data = mx.random.normal((B, H, N, D)).astype(dtype)

        k_packed, k_norms = turboquant_pack_4bit(k_data)
        v_packed, v_norms = turboquant_pack_4bit(v_data)
        k_norms = k_norms.astype(dtype)
        v_norms = v_norms.astype(dtype)
        mx.eval(q, k_packed, k_norms, v_packed, v_norms)

        # Dense reference
        ref = mx.fast.lut_scaled_dot_product_attention(
            q, k_packed, k_norms, v_packed, v_norms,
            ck, cv, scale=scale, bits=4,
            sparse_v_threshold=0.0, sparse_v_mode=0)
        mx.eval(ref)

        # Dense baseline timing
        t_dense = benchmark_fn(
            lambda: mx.fast.lut_scaled_dot_product_attention(
                q, k_packed, k_norms, v_packed, v_norms,
                ck, cv, scale=scale, bits=4,
                sparse_v_threshold=0.0, sparse_v_mode=0),
            iters=args.iters)

        print(f"\n  Dense baseline: {t_dense:.3f}ms")
        print()

        header = (f"  {'Threshold':>12s}  {'Mode':>8s}  "
                  f"{'Latency':>8s}  {'Speedup':>8s}  {'CosSim':>8s}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for thr in thresholds:
            for mode in [0, 1, 2]:
                def fn(t=thr, m=mode):
                    return mx.fast.lut_scaled_dot_product_attention(
                        q, k_packed, k_norms, v_packed, v_norms,
                        ck, cv, scale=scale, bits=4,
                        sparse_v_threshold=t, sparse_v_mode=m)

                out = fn()
                mx.eval(out)
                cs = cossim(ref, out)
                t_ms = benchmark_fn(fn, iters=args.iters)
                speedup = t_dense / t_ms

                thr_str = f"{thr:.0e}" if thr > 0 else "0"
                print(f"  {thr_str:>12s}  {MODE_NAMES[mode]:>8s}  "
                      f"{t_ms:>7.3f}ms  {speedup:>7.3f}x  {cs:>8.5f}")

            if thr > 0:
                print()


def run_real_model_bench(args):
    """Benchmark with real model KV cache patterns."""
    print("=" * 80)
    print(f"REAL MODEL SPARSE-V BENCHMARK: {args.model}")
    print("=" * 80)
    print()

    from mlx_lm import load

    model, tokenizer = load(args.model)
    lm = model.language_model

    fa_layers = [i for i, l in enumerate(lm.model.layers)
                 if hasattr(l, 'self_attn')]
    attn0 = lm.model.layers[fa_layers[0]].self_attn
    n_q = attn0.num_attention_heads
    n_kv = attn0.num_key_value_heads
    D = attn0.head_dim
    scale = attn0.scale

    dtype = mx.float32 if D == 256 else mx.bfloat16
    print(f"Layers: {len(fa_layers)}  n_q={n_q}  n_kv={n_kv}  D={D}")
    print(f"scale={scale}  dtype={dtype}")

    ck = make_centroids(D, dtype=dtype)
    cv = make_centroids(D, dtype=dtype)

    prompt = (
        "Explain the theory of general relativity in detail, covering spacetime "
        "curvature, the equivalence principle, gravitational time dilation. "
        "The quick brown fox jumps over the lazy dog. "
        "To be or not to be, that is the question. "
    )

    thresholds = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2]

    for ctx in args.ctx:
        base = tokenizer.encode(prompt)
        if len(base) < ctx:
            toks = (base * ((ctx // len(base)) + 1))[:ctx]
        else:
            toks = base[:ctx]
        actual = len(toks)

        print(f"\n{'_' * 78}")
        print(f"Prefilling {actual} tokens...")

        inp = mx.array([toks])
        cache = lm.make_cache()
        out = lm(inp, cache=cache)
        mx.eval(out)

        # Use first FA layer
        idx = fa_layers[0]
        c = cache[idx]
        seq = c.offset

        k_data = c.keys[:, :, :seq, :].astype(dtype)
        v_data = c.values[:, :, :seq, :].astype(dtype)
        q_data = mx.random.normal((1, n_q, 1, D)).astype(dtype)
        mx.eval(k_data, v_data, q_data)

        # Pack K and V
        k_packed, k_norms = turboquant_pack_4bit(k_data)
        v_packed, v_norms = turboquant_pack_4bit(v_data)
        k_norms = k_norms.astype(dtype)
        v_norms = v_norms.astype(dtype)
        mx.eval(k_packed, k_norms, v_packed, v_norms)

        # FP reference
        fp_fn = lambda: mx.fast.scaled_dot_product_attention(
            q_data, k_data, v_data, scale=scale)
        out_fp = fp_fn()
        mx.eval(out_fp)
        t_fp = benchmark_fn(fp_fn, iters=args.iters)

        # Dense LUT reference
        def dense_fn():
            return mx.fast.lut_scaled_dot_product_attention(
                q_data, k_packed, k_norms, v_packed, v_norms,
                ck, cv, scale=scale, bits=4,
                sparse_v_threshold=0.0, sparse_v_mode=0)
        out_dense = dense_fn()
        mx.eval(out_dense)
        t_dense = benchmark_fn(dense_fn, iters=args.iters)

        cos_dense_fp = cossim(out_fp, out_dense)
        print(f"  N={actual}  FP={t_fp:.3f}ms  Dense_LUT={t_dense:.3f}ms  "
              f"cos(FP,LUT)={cos_dense_fp:.5f}")
        print()

        header = (f"  {'Threshold':>12s}  {'Mode':>8s}  "
                  f"{'Latency':>8s}  {'vs Dense':>8s}  {'vs FP':>8s}  "
                  f"{'CosSim':>8s}  {'Sparsity':>8s}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for thr in thresholds:
            sparsity = compute_attention_sparsity(
                q_data, k_data, scale, max(thr, 1e-10))

            for mode in [1, 2]:  # Skip mode 0 (same as dense at thr=0)
                def fn(t=thr, m=mode):
                    return mx.fast.lut_scaled_dot_product_attention(
                        q_data, k_packed, k_norms, v_packed, v_norms,
                        ck, cv, scale=scale, bits=4,
                        sparse_v_threshold=t, sparse_v_mode=m)

                out = fn()
                mx.eval(out)
                cs = cossim(out_dense, out)
                t_ms = benchmark_fn(fn, iters=args.iters)
                sp_dense = t_dense / t_ms
                sp_fp = t_fp / t_ms

                thr_str = f"{thr:.0e}" if thr > 0 else "0"
                print(f"  {thr_str:>12s}  {MODE_NAMES[mode]:>8s}  "
                      f"{t_ms:>7.3f}ms  {sp_dense:>7.3f}x  {sp_fp:>7.3f}x  "
                      f"{cs:>8.5f}  {sparsity*100:>7.1f}%")

            if thr > 0:
                print()

        del cache, out


def main():
    parser = argparse.ArgumentParser(
        description="Sparse-V optimization benchmark for LUT SDPA kernel")
    parser.add_argument("--real-model", action="store_true",
                        help="Use real model KV cache instead of synthetic data")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit",
                        help="Model to use for real-model benchmark")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="Head dimension for synthetic benchmark")
    parser.add_argument("--n-heads", type=int, default=32,
                        help="Number of heads for synthetic benchmark")
    parser.add_argument("--seq-lengths", type=int, nargs="+",
                        default=[512, 2048, 8192],
                        help="Sequence lengths to test")
    parser.add_argument("--ctx", type=int, nargs="+",
                        default=[512, 2048, 8192],
                        help="Context lengths for real-model benchmark")
    parser.add_argument("--iters", type=int, default=50,
                        help="Number of benchmark iterations")
    args = parser.parse_args()

    print(f"MLX version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    print()

    if args.real_model:
        run_real_model_bench(args)
    else:
        run_synthetic_bench(args)


if __name__ == "__main__":
    main()
