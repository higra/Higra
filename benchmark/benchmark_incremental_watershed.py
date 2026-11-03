"""
Quick benchmark: Incremental Seeded Watershed vs. Classical Seeded Watershed.

Strategy mirrors the randomized test in test/cpp/algo/test_watershed.cpp:
  - 4-adjacency grid graph of size SIZE x SIZE
  - Random L1 edge weights from a random flat image
  - Interleaved add / remove seed batches (same probabilities as the C++ test)
  - Timings recorded at every interaction (add or remove batch)

Metrics plotted:
  - Time per interaction (both methods)
  - Preprocessing time (0 for classical, BPT build for incremental)
  - Average time per interaction
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import higra as hg

# ── parameters ──────────────────────────────────────────────────────────────
SIZE = 1000          # grid side length  (SIZE x SIZE vertices)
NUM_STEPS = 80      # number of add/remove interactions
NUM_IMAGES = 1      # number of independent random images to average over
MARKER_RADIUS = 1   # square brush radius (same as C++ test)
NUM_SEED_REMOVED = 4
RNG_SEED = 42
NUKE_PROBA = 0
# ────────────────────────────────────────────────────────────────────────────


def run_one_image(image_array, g, size, num_steps, rng):
    """
    Run one full simulation on a single image.
    Returns (times_incr, times_classic, preproc_incr) — all in seconds.
    times_* are per-step wall-clock timings.
    """
    num_vertices = size * size
    edge_weights = hg.weight_graph(g, image_array, hg.WeightFunction.L1)

    # ── Incremental preprocessing ────────────────────────────────────────
    t0 = time.perf_counter()
    iws = hg.IncrementalWatershedCut(g, edge_weights)
    preproc_incr = time.perf_counter() - t0

    # shared seed state
    seeds_flat = np.zeros(num_vertices, dtype=np.int64)
    seed_dic = {}          # vertex -> label

    times_incr = []
    times_classic = []

    for step in range(num_steps):
        maybe_add = (len(seed_dic) == 0) or (rng.random() >= 0.3)

        if maybe_add:
            label = rng.randint(1, 9)
            x = rng.randint(0, size - 1)
            y = rng.randint(0, size - 1)

            new_verts = []
            new_labels = []
            for dy in range(-MARKER_RADIUS, MARKER_RADIUS + 1):
                for dx in range(-MARKER_RADIUS, MARKER_RADIUS + 1):
                    nx_, ny_ = x + dx, y + dy
                    if 0 <= nx_ < size and 0 <= ny_ < size:
                        v = ny_ * size + nx_
                        if v not in seed_dic:
                            seeds_flat[v] = label
                            seed_dic[v] = label
                            new_verts.append(v)
                            new_labels.append(label)

            if not new_verts:
                continue

            sv = np.array(new_verts, dtype=np.int64)
            sl = np.array(new_labels, dtype=np.int64)

            # Incremental
            t0 = time.perf_counter()
            iws.add_seeds(sv, sl)
            _ = iws.get_labeling()
            times_incr.append(time.perf_counter() - t0)

            # Classical
            t0 = time.perf_counter()
            _ = hg.labelisation_seeded_watershed(
                g, edge_weights, seeds_flat.reshape(size, size))
            times_classic.append(time.perf_counter() - t0)

        else:
            nuke = rng.random() > 1 - NUKE_PROBA
            if nuke:
                removed = list(seed_dic.keys())
                seeds_flat[removed] = 0
                seed_dic.clear()
            else:
                keys = list(seed_dic.keys())
                rng.shuffle(keys)
                n_remove = min(len(keys), NUM_SEED_REMOVED)
                removed = keys[:n_remove]
                for v in removed:
                    seeds_flat[v] = 0
                    del seed_dic[v]

            if not removed:
                continue

            sv = np.array(removed, dtype=np.int64)

            # Incremental
            t0 = time.perf_counter()
            iws.remove_seeds(sv)
            _ = iws.get_labeling()
            times_incr.append(time.perf_counter() - t0)

            # Classical
            t0 = time.perf_counter()
            _ = hg.labelisation_seeded_watershed(
                g, edge_weights, seeds_flat.reshape(size, size))
            times_classic.append(time.perf_counter() - t0)

    return times_incr, times_classic, preproc_incr


def main():
    rng = random.Random(RNG_SEED)
    g = hg.get_4_adjacency_graph((SIZE, SIZE))
    num_vertices = SIZE * SIZE

    all_times_incr = []
    all_times_classic = []
    all_preproc = []

    print(f"Benchmarking on {NUM_IMAGES} image(s), {NUM_STEPS} steps each …")
    for img_idx in range(NUM_IMAGES):
        np_rng = np.random.default_rng(RNG_SEED + img_idx)
        image = np_rng.random(num_vertices)

        ti, tc, tp = run_one_image(image, g, SIZE, NUM_STEPS, rng)
        all_times_incr.append(ti)
        all_times_classic.append(tc)
        all_preproc.append(tp)
        print(f"  image {img_idx + 1}/{NUM_IMAGES}: "
              f"{len(ti)} interactions, "
              f"preproc_incr={tp*1e3:.2f} ms, "
              f"avg_incr={np.mean(ti)*1e3:.2f} ms, "
              f"avg_classic={np.mean(tc)*1e3:.2f} ms")

    # ── Aggregate ────────────────────────────────────────────────────────
    # Pad to same length for averaging across images
    max_len = max(len(t) for t in all_times_incr)

    def pad(lst, length):
        return lst + [float("nan")] * (length - len(lst))

    arr_incr = np.array([pad(t, max_len) for t in all_times_incr])   # (images, steps)
    arr_classic = np.array([pad(t, max_len) for t in all_times_classic])

    mean_incr = np.nanmean(arr_incr, axis=0) * 1e3     # ms
    mean_classic = np.nanmean(arr_classic, axis=0) * 1e3
    mean_preproc_incr = np.mean(all_preproc) * 1e3
    # Classical has no preprocessing
    mean_preproc_classic = 0.0

    avg_incr = np.nanmean(arr_incr) * 1e3
    avg_classic = np.nanmean(arr_classic) * 1e3

    steps = np.arange(1, max_len + 1)

    print(f"\nResults (averaged over {NUM_IMAGES} image(s)):")
    print(f"  Incremental  — preproc: {mean_preproc_incr:.2f} ms, "
          f"avg per interaction: {avg_incr:.2f} ms")
    print(f"  Classical    — preproc: {mean_preproc_classic:.2f} ms, "
          f"avg per interaction: {avg_classic:.2f} ms")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle(
        f"Incremental vs Classical Seeded Watershed  —  {SIZE}×{SIZE} grid, "
        f"{NUM_IMAGES} image(s) averaged",
        fontsize=12)

    # (1) Time per interaction
    ax = axes[0]
    ax.plot(steps, mean_incr, label="Incremental", color="tab:blue")
    ax.plot(steps, mean_classic, label="Classical", color="tab:orange")
    ax.set_xlabel("Interaction #")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Time per interaction")
    ax.legend()
    ax.grid(True, alpha=0.4)

    # (2) Preprocessing time
    #ax = axes[1]
    methods = ["Incremental", "Classical"]
    preprocs = [mean_preproc_incr, mean_preproc_classic]
    #bars = ax.bar(methods, preprocs, color=["tab:blue", "tab:orange"])
    #ax.set_ylabel("Time (ms)")
    #ax.set_title("Preprocessing time")
    #ax.grid(True, alpha=0.4, axis="y")
    #for bar, val in zip(bars, preprocs):
    #    ax.text(bar.get_x() + bar.get_width() / 2,
    #            bar.get_height() + 0.01 * max(preprocs + [1e-6]),
    #            f"{val:.2f} ms", ha="center", va="bottom", fontsize=9)

    # (3) Average time per interaction
    ax = axes[1]
    avgs = [avg_incr, avg_classic]
    bars = ax.bar(methods, avgs, color=["tab:blue", "tab:orange"])
    ax.set_ylabel("Time (ms)")
    ax.set_title("Average time per interaction")
    ax.grid(True, alpha=0.4, axis="y")
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(avgs + [1e-6]),
                f"{val:.2f} ms", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = "watershed_benchmark.png"
    plt.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()

