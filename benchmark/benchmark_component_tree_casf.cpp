/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Wonder Alexandre Luz Alves                              *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "higra/algo/tree.hpp"
#include "higra/attribute/tree_attribute.hpp"
#include "higra/hierarchy/component_tree.hpp"
#include "higra/hierarchy/component_tree_casf.hpp"
#include "higra/image/graph_image.hpp"

using namespace hg;

namespace {

/*
 * CASF benchmark overview
 * -----------------------
 *
 * This benchmark compares the CASF implementation with the naive alternating
 * max-tree/min-tree-by-area reference on synthetic grayscale images.
 *
 * Default synthetic mode
 * - One benchmark case corresponds to one generated grayscale image, identified
 *   by model, resolution, threshold count, and seed index.
 * - natural_like combines multi-scale value noise, broad illumination, and
 *   soft ridge-like structures to produce irregular natural-looking variation.
 * - piecewise_scene combines smooth backgrounds, soft geometric objects, and
 *   weak local texture to produce piecewise-smooth scene-like content.
 * - Current default profile: 10 seeds per model, 640x480, 1 repetition,
 *   0 warmup, threshold counts {2, 4, 8, 16, 32}.
 *
 * Threshold schedule
 * - The fixed threshold sequence is:
 *   10 30 40 59 69 89 99 119 138 148 168 178 198 217 227 247
 *   257 277 286 306 326 336 356 365 385 405 415 435 444 464 474 494
 * - If a case requests n thresholds, the benchmark uses the first n values of
 *   this sequence exactly.
 *
 * Validation
 * - Every case validates the CASF output against the naive reference before
 *   entering the measured loop.
 * - speedup_vs_naive > 1 means CASF is faster than the naive baseline.
 *
 * How to run
 * - CASF plus naive reference:
 *   ./build-benchmark/benchmark/benchmark_higra \
 *     --benchmark_filter='BM_component_tree_casf_.*' \
 *     --benchmark_counters_tabular=true
 * 
 * - CASF only:
 *   ./build-benchmark/benchmark/benchmark_higra \
 *     --benchmark_filter='BM_component_tree_casf_area/.*' \
 *     --benchmark_counters_tabular=true
 *
 * Representative results on Apple Silicon (640x480, 10 seeds, 1 repetition)
 * - natural_like:
 *   thresholds=2  -> about 58-69 ms, speedup about 1.50-1.73
 *   thresholds=4  -> about 61-72 ms, speedup about 2.78-3.18
 *   thresholds=8  -> about 64-74 ms, speedup about 5.29-6.07
 *   thresholds=16 -> about 70-78 ms, speedup about 9.95-11.30
 *   thresholds=32 -> about 78-84 ms, speedup about 15.94-20.16
 * - piecewise_scene:
 *   thresholds=2  -> about 44-48 ms, speedup about 1.89-1.96
 *   thresholds=4  -> about 45-48 ms, speedup about 3.32-3.87
 *   thresholds=8  -> about 45-51 ms, speedup about 7.21-8.00
 *   thresholds=16 -> about 45-48 ms, speedup about 14.20-15.29
 *   thresholds=32 -> about 46-50 ms, speedup about 28.29-31.45
 *
 */


enum class BenchmarkImageModel {
    natural_like,
    piecewise_scene
};

constexpr std::array<uint32_t, 10> kBenchmarkSeeds = {
        0x13579u,
        0x24680u,
        0x35791u,
        0x46802u,
        0x57913u,
        0x68A24u,
        0x79B35u,
        0x8AC46u,
        0x9BD57u,
        0xACE68u
};
constexpr int kBenchmarkRepetitions = 1;
constexpr double kBenchmarkWarmupSeconds = 0.0;
constexpr int64_t kBenchmarkNumCols = 640;
constexpr int64_t kBenchmarkNumRows = 480;

const char *benchmark_image_model_name(BenchmarkImageModel model) {
    switch (model) {
        case BenchmarkImageModel::natural_like:
            return "natural_like";
        case BenchmarkImageModel::piecewise_scene:
            return "piecewise_scene";
    }
    return "unknown";
}

// Builds the common textual prefix used in synthetic benchmark labels.
std::string benchmark_case_label(BenchmarkImageModel model, int numThresholds) {
    return std::string(benchmark_image_model_name(model))
           + ", thresholds=" + std::to_string(numThresholds)
           + ", seeds=" + std::to_string(kBenchmarkSeeds.size());
}

// Converts a floating-point intensity to uint8 with saturation.
uint8_t clamp_to_u8(double value) {
    return (uint8_t) std::clamp<int>((int) std::lround(value), 0, 255);
}

// Small deterministic hash used as the base pseudo-random source for procedural textures.
double benchmark_hash01(uint32_t y, uint32_t x, uint32_t seed) {
    uint32_t h = seed;
    h ^= y + 0x9e3779b9u + (h << 6) + (h >> 2);
    h ^= x + 0x9e3779b9u + (h << 6) + (h >> 2);
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return (double) (h & 0x00ffffffu) / (double) 0x01000000u;
}

// Bilinearly interpolated value noise at a given scale.
double benchmark_value_noise(double y, double x, double scale, uint32_t seed) {
    const auto smoothstep01 = [](double t) {
        return t * t * (3.0 - 2.0 * t);
    };
    const double ys = y / scale;
    const double xs = x / scale;
    const auto y0 = (uint32_t) std::floor(ys);
    const auto x0 = (uint32_t) std::floor(xs);
    const auto y1 = y0 + 1u;
    const auto x1 = x0 + 1u;
    const double fy = smoothstep01(ys - std::floor(ys));
    const double fx = smoothstep01(xs - std::floor(xs));

    const double v00 = benchmark_hash01(y0, x0, seed);
    const double v01 = benchmark_hash01(y0, x1, seed);
    const double v10 = benchmark_hash01(y1, x0, seed);
    const double v11 = benchmark_hash01(y1, x1, seed);

    const double vx0 = v00 + fx * (v01 - v00);
    const double vx1 = v10 + fx * (v11 - v10);
    return vx0 + fy * (vx1 - vx0);
}

// Four-octave fractional Brownian motion used to produce natural-looking low-frequency structure.
double benchmark_fbm(double y, double x,
                     const std::array<double, 4> &scales,
                     const std::array<uint32_t, 4> &seeds) {
    double sum = 0.0;
    double amplitude = 1.0;
    double norm = 0.0;
    for (std::size_t i = 0; i < scales.size(); ++i) {
        sum += amplitude * benchmark_value_noise(y, x, scales[i], seeds[i]);
        norm += amplitude;
        amplitude *= 0.5;
    }
    return sum / norm;
}

// Generic smoothstep between two edges, reused by soft geometric masks.
double benchmark_smoothstep(double edge0, double edge1, double x) {
    const double t = std::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Soft disk indicator with feathered boundary.
double benchmark_soft_disk(double x, double y, double cx, double cy, double radius, double feather) {
    const double d = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
    return 1.0 - benchmark_smoothstep(radius - feather, radius + feather, d);
}

// Soft ellipse indicator with feathered boundary.
double benchmark_soft_ellipse(double x, double y, double cx, double cy, double rx, double ry, double feather) {
    const double dx = (x - cx) / rx;
    const double dy = (y - cy) / ry;
    const double d = std::sqrt(dx * dx + dy * dy);
    const double normalizedFeather = feather / std::max(rx, ry);
    return 1.0 - benchmark_smoothstep(1.0 - normalizedFeather, 1.0 + normalizedFeather, d);
}

// Soft axis-aligned rectangle indicator with feathered boundary.
double benchmark_soft_rect(double x, double y, double x0, double y0, double x1, double y1, double feather) {
    const double left = benchmark_smoothstep(x0 - feather, x0 + feather, x);
    const double right = 1.0 - benchmark_smoothstep(x1 - feather, x1 + feather, x);
    const double top = benchmark_smoothstep(y0 - feather, y0 + feather, y);
    const double bottom = 1.0 - benchmark_smoothstep(y1 - feather, y1 + feather, y);
    return left * right * top * bottom;
}

// Natural-like synthetic pattern: irregular smooth texture, broad illumination, and soft ridges.
array_1d<uint8_t> make_natural_like_benchmark_image(index_t numRows, index_t numCols, uint32_t seed) {
    const index_t numPixels = numRows * numCols;
    auto image = array_1d<uint8_t>::from_shape({(size_t) numPixels});
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit01(0.0, 1.0);
    std::uniform_real_distribution<double> ampJitter(0.8, 1.25);
    std::uniform_real_distribution<double> freqJitter(0.75, 1.35);
    std::uniform_real_distribution<double> angleJitter(-M_PI, M_PI);
    std::uniform_real_distribution<double> offsetJitter(-0.18, 0.18);

    const int backgroundFamily = (int) (rng() % 3u);
    const int ridgeCount = 2 + (int) (rng() % 3u);
    const std::array<double, 4> fbmScales = {
            36.0 * freqJitter(rng),
            18.0 * freqJitter(rng),
            9.0 * freqJitter(rng),
            4.5 * freqJitter(rng)};
    const std::array<uint32_t, 4> fbmSeeds = {
            seed + 0x12345u, seed + 0x23456u, seed + 0x34567u, seed + 0x45678u};
    const double phase1 = 2.0 * M_PI * unit01(rng);
    const double phase2 = 2.0 * M_PI * unit01(rng);
    const double phase3 = 2.0 * M_PI * unit01(rng);
    const double phase4 = 2.0 * M_PI * unit01(rng);
    const double edgeBias = 2.0 * unit01(rng) - 1.0;
    const double textureLargeAmplitude = 96.0 * ampJitter(rng);
    const double textureSmallScale = 4.0 * freqJitter(rng);
    const double orientedFreqX1 = 0.020 + 0.030 * unit01(rng);
    const double orientedFreqY1 = 0.025 + 0.040 * unit01(rng);
    const double orientedFreqX2 = 0.018 + 0.035 * unit01(rng);
    const double orientedFreqY2 = 0.016 + 0.028 * unit01(rng);
    const double dir1 = angleJitter(rng);
    const double dir2 = angleJitter(rng);
    const double ux1 = std::cos(dir1);
    const double uy1 = std::sin(dir1);
    const double ux2 = std::cos(dir2);
    const double uy2 = std::sin(dir2);
    const double centerX = 0.45 + offsetJitter(rng);
    const double centerY = 0.52 + offsetJitter(rng);
    const double notchDx = 0.18 + 0.12 * unit01(rng);
    const double notchDy = -0.24 + 0.16 * unit01(rng);
    std::vector<std::array<double, 5>> ridges;
    ridges.reserve((size_t) ridgeCount);
    for (int i = 0; i < ridgeCount; ++i) {
        ridges.push_back({
                unit01(rng),
                unit01(rng),
                0.10 + 0.18 * unit01(rng),
                0.04 + 0.08 * unit01(rng),
                (unit01(rng) < 0.5 ? -1.0 : 1.0) * (18.0 + 24.0 * unit01(rng))
        });
    }

    for (index_t r = 0; r < numRows; ++r) {
        for (index_t c = 0; c < numCols; ++c) {
            const index_t p = r * numCols + c;
            const double y = (double) r;
            const double x = (double) c;
            const double xn = x / (double) std::max<index_t>(1, numCols - 1);
            const double yn = y / (double) std::max<index_t>(1, numRows - 1);
            const double coord1 = ux1 * xn + uy1 * yn;
            const double coord2 = ux2 * xn + uy2 * yn;

            double illumination = 70.0;
            if (backgroundFamily == 0) {
                illumination += 42.0 * coord1 + 26.0 * coord2;
            } else if (backgroundFamily == 1) {
                illumination += 30.0 * std::sin(4.8 * coord1 + phase1) + 24.0 * std::cos(3.7 * coord2 + phase2);
            } else {
                const double dx = xn - centerX;
                const double dy = yn - centerY;
                illumination += 110.0 * std::exp(-(dx * dx / 0.22 + dy * dy / 0.15));
                illumination -= 36.0 * std::exp(-((dx - notchDx) * (dx - notchDx) / 0.08 + (dy - notchDy) * (dy - notchDy) / 0.05));
            }

            const double textureLarge = textureLargeAmplitude * benchmark_fbm(0.85 * y, 0.85 * x, fbmScales, fbmSeeds);
            const double textureSmall = 18.0 * benchmark_value_noise(y, x, textureSmallScale, seed + 0x56789u);
            const double oriented =
                    12.0 * std::sin(orientedFreqX1 * x + orientedFreqY1 * y + phase3)
                    + 9.0 * std::cos(orientedFreqX2 * x - orientedFreqY2 * y + phase4);

            const double edgeLike =
                    benchmark_value_noise(y, x, 19.0, seed + 0x6789Au) > 0.5 ? (9.0 + 3.0 * edgeBias) : (-9.0 - 3.0 * edgeBias);

            double ridgeContribution = 0.0;
            for (const auto &ridge: ridges) {
                const double cx = ridge[0] * (double) numCols;
                const double cy = ridge[1] * (double) numRows;
                const double rx = ridge[2] * (double) numCols;
                const double ry = ridge[3] * (double) numRows;
                ridgeContribution += ridge[4] * benchmark_soft_ellipse(x, y, cx, cy, rx, ry, 6.0);
            }

            image(p) = clamp_to_u8(illumination + textureLarge + textureSmall + oriented + edgeLike + ridgeContribution - 55.0);
        }
    }
    return image;
}

// Piecewise-scene synthetic pattern: smooth background plus soft geometric objects and weak texture.
array_1d<uint8_t> make_piecewise_scene_benchmark_image(index_t numRows, index_t numCols, uint32_t seed) {
    const index_t numPixels = numRows * numCols;
    auto image = array_1d<uint8_t>::from_shape({(size_t) numPixels});
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit01(0.0, 1.0);
    const std::array<double, 4> backgroundScales = {96.0, 48.0, 24.0, 12.0};
    const std::array<uint32_t, 4> backgroundSeeds = {
            seed + 0x13579u, seed + 0x24680u, seed + 0x35791u, seed + 0x46802u};
    const std::array<double, 4> objectScales = {40.0, 20.0, 10.0, 5.0};
    const std::array<uint32_t, 4> objectSeeds = {
            seed + 0x51051u, seed + 0x62062u, seed + 0x73073u, seed + 0x84084u};
    const int backgroundFamily = (int) (rng() % 3u);
    const int objectCount = 4 + (int) (rng() % 4u);
    struct ObjectSpec {
        int type;
        double cx;
        double cy;
        double sx;
        double sy;
        double feather;
        double contrast;
        double phase;
    };
    std::vector<ObjectSpec> objects;
    objects.reserve((size_t) objectCount);
    for (int i = 0; i < objectCount; ++i) {
        const int type = (int) (rng() % 5u);
        const double cx = (0.12 + 0.76 * unit01(rng)) * (double) numCols;
        const double cy = (0.12 + 0.76 * unit01(rng)) * (double) numRows;
        const double sx = (0.08 + 0.20 * unit01(rng)) * (double) numCols;
        const double sy = (0.06 + 0.22 * unit01(rng)) * (double) numRows;
        const double feather = 2.5 + 4.5 * unit01(rng);
        const double contrast = (unit01(rng) < 0.35 ? -1.0 : 1.0) * (34.0 + 70.0 * unit01(rng));
        const double phase = 2.0 * M_PI * unit01(rng);
        objects.push_back({type, cx, cy, sx, sy, feather, contrast, phase});
    }

    for (index_t r = 0; r < numRows; ++r) {
        for (index_t c = 0; c < numCols; ++c) {
            const index_t p = r * numCols + c;
            const double y = (double) r;
            const double x = (double) c;

            const double xn = x / (double) std::max<index_t>(1, numCols - 1);
            const double yn = y / (double) std::max<index_t>(1, numRows - 1);

            double value = 0.0;
            if (backgroundFamily == 0) {
                value = 40.0 + 76.0 * yn + 30.0 * xn;
                value += 18.0 * std::sin(0.90 * xn + 1.35 * yn);
                value += 11.0 * std::cos(1.70 * xn - 1.10 * yn);
            } else if (backgroundFamily == 1) {
                value = 92.0 + 28.0 * std::sin(3.1 * xn + 2.0 * yn) + 36.0 * (benchmark_fbm(0.8 * y, 0.8 * x, backgroundScales, backgroundSeeds) - 0.5);
            } else {
                const double dx = xn - 0.48;
                const double dy = yn - 0.54;
                value = 58.0 + 115.0 * std::exp(-(dx * dx / 0.18 + dy * dy / 0.14));
                value += 20.0 * std::sin(2.8 * xn - 2.2 * yn);
            }

            const double objectTexture = benchmark_fbm(1.2 * y, 1.2 * x, objectScales, objectSeeds) - 0.5;
            const double fineTexture = benchmark_value_noise(y, x, 6.0, seed + 0x97531u) - 0.5;

            for (const auto &obj: objects) {
                double mask = 0.0;
                if (obj.type == 0) {
                    mask = benchmark_soft_disk(x, y, obj.cx, obj.cy, obj.sx, obj.feather);
                } else if (obj.type == 1) {
                    mask = benchmark_soft_ellipse(x, y, obj.cx, obj.cy, obj.sx, obj.sy, obj.feather);
                } else if (obj.type == 2) {
                    mask = benchmark_soft_rect(x, y, obj.cx - obj.sx, obj.cy - obj.sy, obj.cx + obj.sx, obj.cy + obj.sy, obj.feather);
                } else if (obj.type == 3) {
                    const double outer = benchmark_soft_disk(x, y, obj.cx, obj.cy, obj.sx, obj.feather);
                    const double inner = benchmark_soft_disk(x, y, obj.cx, obj.cy, 0.58 * obj.sx, obj.feather);
                    mask = std::max(0.0, outer - inner);
                } else {
                    mask = benchmark_soft_ellipse(x, y, obj.cx, obj.cy, obj.sx, 0.35 * obj.sy, obj.feather);
                }
                value += mask * (obj.contrast + 10.0 * objectTexture + 6.0 * std::sin(0.020 * x + 0.024 * y + obj.phase));
            }

            // Add weak acquisition-like variation without turning the scene into a high-frequency pattern.
            value += 5.0 * fineTexture;
            value += 2.5 * std::sin(0.070 * x + 0.045 * y);

            image(p) = clamp_to_u8(value);
        }
    }

    return image;
}

// Dispatches synthetic image generation according to the selected model.
array_1d<uint8_t> make_benchmark_image(BenchmarkImageModel model, index_t numRows, index_t numCols, uint32_t seed) {
    switch (model) {
        case BenchmarkImageModel::natural_like:
            return make_natural_like_benchmark_image(numRows, numCols, seed);
        case BenchmarkImageModel::piecewise_scene:
            return make_piecewise_scene_benchmark_image(numRows, numCols, seed);
    }
    hg_assert(false, "Unknown benchmark image model.");
    return make_natural_like_benchmark_image(numRows, numCols, seed);
}

// Returns the first n values of the fixed benchmark threshold sequence.
std::vector<double> make_area_thresholds(int numThresholds) {
    hg_assert(numThresholds > 0, "Number of thresholds must be positive.");
    constexpr std::array<int, 32> kThresholdPrefixSequence = {
            10, 30, 40, 59, 69, 89, 99, 119,
            138, 148, 168, 178, 198, 217, 227, 247,
            257, 277, 286, 306, 326, 336, 356, 365,
            385, 405, 415, 435, 444, 464, 474, 494
    };
    hg_assert(numThresholds <= (int) kThresholdPrefixSequence.size(),
              "Requested number of thresholds exceeds the configured prefix sequence.");

    std::vector<double> thresholds;
    thresholds.reserve((std::size_t) numThresholds);
    for (int step = 0; step < numThresholds; ++step) {
        const double threshold = (double) kThresholdPrefixSequence[(std::size_t) step];
        if (thresholds.empty() || threshold > thresholds.back()) {
            thresholds.push_back(threshold);
        }
    }
    return thresholds;
}

template<typename array_t>
// Exact image equality check used by pre-benchmark validation.
bool same_image(const array_t &lhs, const array_t &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (index_t i = 0; i < (index_t) lhs.size(); ++i) {
        if (lhs(i) != rhs(i)) {
            return false;
        }
    }
    return true;
}

template<typename image_t, typename graph_t>
// One step of the naive alternating max-tree/min-tree-by-area reference.
image_t apply_naive_threshold_by_area(const graph_t &graph, const image_t &image, double threshold) {
    auto maxTree = component_tree_max_tree(graph, image);
    auto maxArea = attribute_area(maxTree.tree);
    auto filtered = reconstruct_leaf_data(maxTree.tree, maxTree.altitudes, maxArea <= threshold);

    auto minTree = component_tree_min_tree(graph, filtered);
    auto minArea = attribute_area(minTree.tree);
    return reconstruct_leaf_data(minTree.tree, minTree.altitudes, minArea <= threshold);
}

template<typename image_t, typename graph_t>
// Applies the full naive threshold sequence by repeatedly rebuilding both trees.
image_t run_naive_area_sequence(const graph_t &graph, const image_t &image, const std::vector<double> &thresholds) {
    image_t current = image;
    for (double threshold: thresholds) {
        current = apply_naive_threshold_by_area(graph, current, threshold);
    }
    return current;
}

template<typename image_t, typename graph_t>
// Runs CASF once for the complete threshold sequence.
image_t run_casf_area_sequence(const graph_t &graph, const image_t &image, const std::vector<double> &thresholds) {
    ComponentTreeCasf<uint8_t, graph_t> runner(graph, image);
    return runner.filter(thresholds);
}

template<typename Fn>
// Measures wall-clock elapsed time in seconds for a callable.
double time_seconds(Fn &&fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

struct AreaBenchmarkTiming {
    double casfSeconds = 0.0;
    double naiveSeconds = 0.0;
};

// In-memory representation of one synthetic benchmark case.
struct BenchmarkSample {
    std::string label;
    index_t numRows;
    index_t numCols;
    array_1d<uint8_t> image;
};

constexpr std::array<int, 5> kThresholdRounds = {2, 4, 8, 16, 32};

// Generates the (width, height, threshold-count, seed-index) tuples used in synthetic mode.
std::vector<std::array<int64_t, 4>> make_benchmark_argument_sets() {
    std::vector<std::array<int64_t, 4>> args;
    args.reserve(kThresholdRounds.size() * kBenchmarkSeeds.size());
    for (int threshold: kThresholdRounds) {
        for (std::size_t seedIndex = 0; seedIndex < kBenchmarkSeeds.size(); ++seedIndex) {
            args.push_back({kBenchmarkNumCols, kBenchmarkNumRows, threshold, (int64_t) seedIndex});
        }
    }
    return args;
}

// Registers all synthetic argument tuples on a google-benchmark instance.
void apply_small_benchmark_arguments(benchmark::Benchmark *b) {
    std::vector<std::vector<int64_t>> args;
    const auto cases = make_benchmark_argument_sets();
    args.reserve(cases.size());
    for (const auto &c: cases) {
        args.push_back({c[0], c[1], c[2], c[3]});
    }
    for (const auto &a: args) {
        b->Args(a);
    }
}

template<typename image_t, typename graph_t>
// Measures both CASF and naive implementations on the same image/threshold set.
AreaBenchmarkTiming measure_reference_timings(const graph_t &graph, const image_t &image, const std::vector<double> &thresholds) {
    AreaBenchmarkTiming timing;
    timing.casfSeconds = time_seconds([&]() {
        const auto output = run_casf_area_sequence(graph, image, thresholds);
        benchmark::DoNotOptimize(output.data());
    });
    timing.naiveSeconds = time_seconds([&]() {
        const auto output = run_naive_area_sequence(graph, image, thresholds);
        benchmark::DoNotOptimize(output.data());
    });
    return timing;
}

template<typename image_t, typename graph_t>
// Asserts that CASF reproduces exactly the naive reference on a single image.
void validate_against_naive(const graph_t &graph, const image_t &image, const std::vector<double> &thresholds) {
    const auto expected = run_naive_area_sequence(graph, image, thresholds);
    const auto actual = run_casf_area_sequence(graph, image, thresholds);
    hg_assert(same_image(expected, actual), "CASF and naive outputs must match in the benchmark fixture.");
}

// Populates the counters shared by all CASF benchmark cases.
void set_common_counters(benchmark::State &state,
                         double numThresholds,
                         double numPixels,
                         double imageIndex,
                         double speedupVsNaive) {
    state.counters["num_thresholds"] = benchmark::Counter(numThresholds, benchmark::Counter::kDefaults, benchmark::Counter::OneK::kIs1000);
    state.counters["num_images"] = benchmark::Counter(1.0, benchmark::Counter::kDefaults, benchmark::Counter::OneK::kIs1000);
    state.counters["num_pixels"] = benchmark::Counter(numPixels, benchmark::Counter::kDefaults, benchmark::Counter::OneK::kIs1000);
    state.counters["image_index"] = benchmark::Counter(imageIndex, benchmark::Counter::kDefaults, benchmark::Counter::OneK::kIs1000);
    state.counters["speedup_vs_naive"] = benchmark::Counter(speedupVsNaive);
}

// Builds one synthetic benchmark sample from a model, resolution and seed index.
BenchmarkSample make_benchmark_sample(BenchmarkImageModel model, index_t numRows, index_t numCols, std::size_t seedIndex) {
    hg_assert(seedIndex < kBenchmarkSeeds.size(), "Synthetic benchmark seed index out of range.");
    const auto seed = kBenchmarkSeeds[seedIndex];
    return BenchmarkSample{
            std::string(benchmark_image_model_name(model)) + "_seed_" + std::to_string(seedIndex),
            numRows,
            numCols,
            make_benchmark_image(model, numRows, numCols, seed)
    };
}

// Applies the default execution policy shared by all CASF google-benchmark entries.
void configure_benchmark_statistics(benchmark::Benchmark *b) {
    b->Iterations(1)->Repetitions(kBenchmarkRepetitions)->MinWarmUpTime(kBenchmarkWarmupSeconds)->ReportAggregatesOnly();
}

// Synthetic CASF benchmark case: one generated image, one threshold prefix, one seed.
static void BM_component_tree_casf_area(benchmark::State &state, BenchmarkImageModel model) {
    const index_t numCols = state.range(0);
    const index_t numRows = state.range(1);
    const int numThresholds = (int) state.range(2);
    const auto seedIndex = (std::size_t) state.range(3);
    const index_t numPixels = numRows * numCols;
    const auto sample = make_benchmark_sample(model, numRows, numCols, seedIndex);
    const auto graph = get_4_adjacency_implicit_graph({numRows, numCols});
    const auto thresholds = make_area_thresholds(numThresholds);
    const auto timing = measure_reference_timings(graph, sample.image, thresholds);

    validate_against_naive(graph, sample.image, thresholds);
    state.SetLabel(benchmark_case_label(model, numThresholds)
                   + ", resolution=" + std::to_string(numCols) + "x" + std::to_string(numRows)
                   + ", image=" + sample.label);
    set_common_counters(state, (double) numThresholds, (double) numPixels, (double) seedIndex, timing.naiveSeconds / timing.casfSeconds);

    for (auto _: state) {
        const auto output = run_casf_area_sequence(graph, sample.image, thresholds);
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }
}

// Synthetic naive-reference benchmark case matching BM_component_tree_casf_area.
static void BM_component_tree_casf_naive_area(benchmark::State &state, BenchmarkImageModel model) {
    const index_t numCols = state.range(0);
    const index_t numRows = state.range(1);
    const int numThresholds = (int) state.range(2);
    const auto seedIndex = (std::size_t) state.range(3);
    const index_t numPixels = numRows * numCols;
    const auto sample = make_benchmark_sample(model, numRows, numCols, seedIndex);
    const auto graph = get_4_adjacency_implicit_graph({numRows, numCols});
    const auto thresholds = make_area_thresholds(numThresholds);

    validate_against_naive(graph, sample.image, thresholds);
    state.SetLabel(benchmark_case_label(model, numThresholds)
                   + ", resolution=" + std::to_string(numCols) + "x" + std::to_string(numRows)
                   + ", image=" + sample.label);
    set_common_counters(state, (double) numThresholds, (double) numPixels, (double) seedIndex, 1.0);

    for (auto _: state) {
        const auto output = run_naive_area_sequence(graph, sample.image, thresholds);
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }
}

} // namespace

namespace {
bool register_component_tree_casf_benchmarks() {
    benchmark::RegisterBenchmark("BM_component_tree_casf_area/natural_like", BM_component_tree_casf_area,
                                 BenchmarkImageModel::natural_like)
            ->Apply(apply_small_benchmark_arguments)
            ->Apply(configure_benchmark_statistics);
    benchmark::RegisterBenchmark("BM_component_tree_casf_area/piecewise_scene", BM_component_tree_casf_area,
                                 BenchmarkImageModel::piecewise_scene)
            ->Apply(apply_small_benchmark_arguments)
            ->Apply(configure_benchmark_statistics);
    benchmark::RegisterBenchmark("BM_component_tree_casf_naive_area/natural_like", BM_component_tree_casf_naive_area,
                                 BenchmarkImageModel::natural_like)
            ->Apply(apply_small_benchmark_arguments)
            ->Apply(configure_benchmark_statistics);
    benchmark::RegisterBenchmark("BM_component_tree_casf_naive_area/piecewise_scene", BM_component_tree_casf_naive_area,
                                 BenchmarkImageModel::piecewise_scene)
            ->Apply(apply_small_benchmark_arguments)
            ->Apply(configure_benchmark_statistics);
    return true;
}
const bool component_tree_casf_benchmarks_registered = register_component_tree_casf_benchmarks();
} // namespace
