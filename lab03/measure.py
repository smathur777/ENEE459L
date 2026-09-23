from __future__ import annotations

import statistics
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    samples = []
    bench.workload.synchronize()
    for _ in range(repeats):
        start = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end = bench.clock()
        samples.append((end - start) / 1_000_000.0)
    return samples


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    source = (
        f"leading prefix above (1 + {WARMUP_TOL}) x median of the run's second half"
    )
    if len(samples) < 4:
        return unknown(source, "too few samples; at least 4 are required")

    settled = statistics.median(samples[len(samples) // 2 :])
    if settled <= 0:
        return unknown(source, "the settled median must be positive")

    threshold = settled * (1 + WARMUP_TOL)
    discarded = 0
    for sample in samples:
        if sample <= threshold:
            break
        discarded += 1
    return measured(
        discarded,
        source,
        settled_rate_ms=round(settled, 4),
        threshold_ms=round(threshold, 4),
        tolerance=WARMUP_TOL,
        retained=len(samples) - discarded,
    )



def summarize(samples: list[float]) -> dict[str, Any]:
    n = len(samples)
    if not samples:
        return dict.fromkeys(("mean", "std", "min", "max", "p50", "p95", "p99")) | {"n": 0}

    ordered = sorted(samples)
    result = {
        "n": n,
        "mean": round(statistics.fmean(ordered), 4),
        "std": round(statistics.stdev(ordered), 4) if n > 2 else 0.0,
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }
    for q in PERCENTILES:
        h = (n - 1) * q / 100
        i = int(h)
        value = ordered[i] + (h - i) * (ordered[min(i + 1, n - 1)] - ordered[i])
        result[f"p{q}"] = round(value, 4)
    return result

def is_multimodal(samples: list[float]) -> dict[str, Any]:
    source = (
        f"widest trimmed gap >= {MULTIMODAL_GAP_RATIO}x the median gap, "
        f"with >= {MIN_MODE_FRACTION:.0%} of samples on each side"
    )
    n = len(samples)
    if n < MIN_SAMPLES_FOR_MODALITY:
        return unknown(source, f"not enough samples; at least {MIN_SAMPLES_FOR_MODALITY} are required")

    ordered = sorted(samples)
    trim = int(n * 0.05)
    trimmed = ordered[trim : n - trim]
    gaps = [right - left for left, right in zip(trimmed, trimmed[1:])]
    typical_gap = statistics.median(gaps)
    if typical_gap <= 0:
        return unknown(source, "timer resolution is too coarse; the median gap is not positive")

    widest_gap = max(gaps)
    ratio = widest_gap / typical_gap
    split = trim + gaps.index(widest_gap) + 1
    groups = (ordered[:split], ordered[split:])
    return measured(
        ratio >= MULTIMODAL_GAP_RATIO
        and all(len(group) >= n * MIN_MODE_FRACTION for group in groups),
        source,
        gap_ratio=round(ratio, 2),
        widest_gap_ms=round(widest_gap, 4),
        typical_gap_ms=round(typical_gap, 5),
        modes=[
            {
                "n": len(group),
                "share": round(len(group) / n, 4),
                "median_ms": round(statistics.median(group), 4),
            }
            for group in groups
        ],
    )

# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    result = bench.runner(["nvpmodel", "-q"])
    if not result.ok or result.returncode != 0:
        return unknown(
            result.source,
            result.error or result.stdout.strip()
            or f"command exited with code {result.returncode}; check nvpmodel permissions",
        )

    findings = unknown(result.source, "NV Power Mode was not found in command output")
    lines = result.stdout.splitlines()
    for i, line in enumerate(lines):
        if "NV Power Mode:" in line:
            mode = line.split("NV Power Mode:", 1)[1].strip()
            try:
                mode_index = int(lines[i + 1].strip())
            except (IndexError, ValueError):
                findings = unknown(result.source, "power mode ID is missing or invalid")
            else:
                if mode:
                    findings = measured(mode, result.source, mode_index=mode_index)
            break

    minimum = read_text(bench.telemetry, CPUFREQ_MIN)
    maximum = read_text(bench.telemetry, CPUFREQ_MAX)
    clock_source = f"{CPUFREQ_MIN} vs {CPUFREQ_MAX}"
    findings["jetson_clocks"] = None
    if minimum is None or maximum is None:
        findings["jetson_clocks_source"] = unknown(clock_source, "CPU frequency limits could not be read")
    else:
        try:
            findings["jetson_clocks"] = int(minimum) == int(maximum)
        except ValueError:
            findings["jetson_clocks_source"] = unknown(clock_source, "CPU frequency limits are not integers")
        else:
            findings["jetson_clocks_source"] = measured(
                f"scaling_min_freq={minimum}, scaling_max_freq={maximum}", clock_source
            )
    return findings



def probe_telemetry(bench: Bench) -> dict[str, Any]:
    temperature_source = f"{THERMAL_ZONES}/*/temp"
    temperatures = []
    try:
        zones = sorted((bench.telemetry / THERMAL_ZONES).glob("thermal_zone*"))
    except OSError:
        zones = []
    for zone in zones:
        relative = f"{THERMAL_ZONES}/{zone.name}"
        raw = read_text(bench.telemetry, f"{relative}/temp")
        if raw is None:
            continue
        try:
            millidegrees = int(raw)
        except ValueError:
            continue
        if millidegrees <= -1000:
            continue
        name = read_text(bench.telemetry, f"{relative}/type") or zone.name
        temperatures.append((millidegrees / 1000.0, name))

    if temperatures:
        hottest, name = max(temperatures, key=lambda reading: reading[0])
        temperature = measured(
            hottest, temperature_source, zone=name, zones_read=len(temperatures)
        )
    else:
        temperature = unknown(temperature_source, "no valid thermal zone temperatures could be read")

    power_reading = read_first(bench.telemetry, POWER_RAIL_CANDIDATES)
    if power_reading is None:
        power = unknown(
            " | ".join(POWER_RAIL_CANDIDATES),
            "none of the documented INA3221 rail paths could be read",
        )
    else:
        source, raw = power_reading
        try:
            power = measured(int(raw), source)
        except ValueError:
            power = unknown(source, "power reading is not an integer")

    gpu_reading = read_first(bench.telemetry, GPU_LOAD_CANDIDATES)
    if gpu_reading is None:
        gpu = unknown(" | ".join(GPU_LOAD_CANDIDATES), "no GPU load file could be read")
    else:
        source, raw = gpu_reading
        try:
            gpu = measured(int(raw) / 10.0, source, units="per-mille / 10")
        except ValueError:
            gpu = unknown(source, "GPU load reading is not an integer")

    return {"temperature_c": temperature, "power_mw": power, "gpu_utilization_percent": gpu}

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

def is_stationary(samples: list[float]) -> dict[str, Any]:
    source = "first vs last third median drift relative to the overall median"
    n = len(samples)
    if n < MIN_SAMPLES_FOR_STATIONARITY:
        return unknown(source, "too few samples to divide into thirds")

    overall_median = statistics.median(samples)
    if overall_median <= 0:
        return unknown(source, "the overall median is not positive")

    k = n // 3
    first_median = statistics.median(samples[:k])
    last_median = statistics.median(samples[-k:])
    drift = last_median - first_median
    relative_drift = abs(drift) / overall_median
    direction = "slower" if drift > 0 else "faster" if drift < 0 else "flat"

    return measured(
        relative_drift <= STATIONARITY_TOL,
        source,
        first_third_median_ms=round(first_median, 4),
        last_third_median_ms=round(last_median, 4),
        drift_ms=round(drift, 4),
        drift_relative=round(relative_drift, 4),
        direction=direction,
        tolerance=STATIONARITY_TOL,
    )
