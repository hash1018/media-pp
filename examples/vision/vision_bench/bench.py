#!/usr/bin/env python3
"""Runs vision_bench over a matrix of configurations, as
docs/benchmarks/vision describes, and prints a Markdown table for each
experiment.

Every configuration runs --reps times, the configurations taking turns
(a, b, c, a, b, c, ...) rather than one after another, so that a GPU
warming up or something else starting on the machine lands on all of them
alike; the table gives the median and the spread. Where nvidia-smi is
found, the GPU's, NVDEC's and NVENC's utilisation and the memory in use are
sampled every 200 ms through each run.

    python3 bench.py --dir ~/opt/bench --arm before=bin/before \
        [--arm after=bin/after] [--reps 3] [--only E0,E4] [--csv results.csv]

An arm is a directory of builds: vision_bench, and optionally
vision_bench-gpu-dcf (built with `--features gpu-dcf`; without it the
GPU-tracking rows are skipped) and model_only (E0's ceiling; without it E0
is skipped), with ONNX Runtime's provider libraries beside them. Given two
arms — media-pp before and after a change — each configuration runs on
both in turn inside the one run, so that both meet the same machine, and
the table sets them side by side. E0 runs on the first arm alone: it does
not go through media-pp.

--dir holds the clips and models the matrix names: people-432p.mp4,
people-1080p.mp4, people-2160p.mp4, yolo11n.onnx, yolo11s.onnx,
yolov10n.onnx and mobilenetv2-12.onnx.
"""

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time

V432, V1080, V2160 = "people-432p.mp4", "people-1080p.mp4", "people-2160p.mp4"
N, S, V10 = "yolo11n.onnx", "yolo11s.onnx", "yolov10n.onnx"
CLS = "mobilenetv2-12.onnx"

# The builds' own names, as Windows gives them.
EXE = ".exe" if os.name == "nt" else ""

# A CPU detector runs a few dozen pictures a second: a few hundred of them
# say as much as the whole file.
# An engine for each model, precision and batch, kept apart: ONNX Runtime
# names one after the graph alone, and two batches in one directory would
# rebuild each other's for minutes on every run.
ENGINES = os.path.expanduser("~/.cache/media-pp/vision-bench-engines")

CPU = ["--backend", "cpu", "--pictures", "400", "--warmup", "40"]
TRT = ["--backend", "tensorrt"]


# E0: the model alone, the ceiling every pipeline below is held against.
MODEL_ONLY = [
    (N, ["--provider", "tensorrt"], "1,2,4,8"),
    (N, ["--provider", "tensorrt", "--fp32"], "1,8"),
    (N, ["--provider", "cuda"], "1,8"),
    (S, ["--provider", "tensorrt"], "1,8"),
    (S, ["--provider", "cuda"], "1"),
    (V10, ["--provider", "tensorrt"], "1"),
    (V10, ["--provider", "cuda"], "1"),
]


def model_only(binary, directory, reps):
    """E0's table: each model, provider and batch, the median of `reps`."""
    seen = {}
    for rep in range(reps):
        for model, args, batches in MODEL_ONLY:
            out = subprocess.run(
                [binary, os.path.join(directory, model), *args, "--batch", batches,
                 "--engine-cache-root", ENGINES],
                capture_output=True, text=True,
            )
            for line in out.stdout.splitlines():
                if line.startswith("MODEL "):
                    r = dict(kv.split("=", 1) for kv in line.split()[1:])
                    key = (r["model"], r["provider"], r["precision"], int(r["batch"]))
                    seen.setdefault(key, []).append(float(r["fps"]))
                    print(f"  {rep} E0 {key}: {r['fps']} fps", file=sys.stderr)
            if out.returncode != 0:
                sys.stderr.write(f"failed: {model} {args}\n{out.stderr}\n")
    print("\n### E0\n")
    print("| model | provider | precision | batch | pictures/s (median) | ms a batch |")
    print("|---|---|---|---:|---:|---:|")
    for (model, provider, precision, batch), fps in seen.items():
        m = statistics.median(fps)
        print(f"| {model} | {provider} | {precision} | {batch} | {m:.0f} | {batch / m * 1e3:.2f} |")


def matrix():
    """(experiment, label, arguments, needs the gpu-dcf build)."""
    m = []

    def add(exp, label, args, gpu_dcf=False):
        m.append((exp, label, args, gpu_dcf))

    # E1: how fast the pictures can come at all.
    for v in (V432, V1080, V2160):
        add("E1", f"nvdec {v}", [v, *TRT])
        add("E1", f"sw-decode {v}", [v, *CPU])

    # E2: who runs the model, at what precision, and which model.
    add("E2", "cpu yolo11n", [V1080, "--model", N, *CPU])
    add("E2", "cuda fp32 yolo11n", [V1080, "--model", N, "--backend", "cuda"])
    add("E2", "tensorrt fp32 yolo11n", [V1080, "--model", N, *TRT, "--fp32"])
    add("E2", "tensorrt fp16 yolo11n", [V1080, "--model", N, *TRT])
    add("E2", "cpu yolo11s", [V1080, "--model", S, *CPU])
    add("E2", "cuda fp32 yolo11s", [V1080, "--model", S, "--backend", "cuda"])
    add("E2", "tensorrt fp16 yolo11s", [V1080, "--model", S, *TRT])
    add("E2", "cuda fp32 yolov10n", [V1080, "--model", V10, "--backend", "cuda"])
    add("E2", "tensorrt fp16 yolov10n", [V1080, "--model", V10, *TRT])

    # E3: the picture's size, which the detector shrinks to 640 anyway.
    for v in (V432, V1080, V2160):
        add("E3", f"tensorrt fp16 {v}", [v, "--model", N, *TRT])

    # E4: several streams, a detector each or one batching them.
    for n in (1, 2, 4, 8):
        add("E4", f"{n} streams, a detector each", [V1080, "--model", N, *TRT, "--streams", str(n)])
        add("E4", f"{n} streams, batched", [V1080, "--model", N, *TRT, "--streams", str(n), "--batch", str(n)])
    for n in (1, 4, 8):
        add("E4", f"{n} streams decode only", [V1080, *TRT, "--streams", str(n)])

    # E5: detecting only some pictures, the tracker filling in the rest.
    for streams, batch in ((1, 0), (4, 4)):
        for interval in (0, 1, 2, 4):
            add(
                "E5",
                f"{streams} stream(s), interval {interval}",
                [V1080, "--model", N, *TRT, "--streams", str(streams), "--batch", str(batch),
                 "--interval", str(interval), "--track", "motion"],
            )

    # E6: what following costs, by motion and by look.
    for interval in (0, 4):
        base = [V1080, "--model", N, *TRT, "--interval", str(interval)]
        add("E6", f"interval {interval}, no tracker", base)
        add("E6", f"interval {interval}, motion", [*base, "--track", "motion"])
        add("E6", f"interval {interval}, visual on the CPU", [*base, "--track", "visual"])
        add("E6", f"interval {interval}, visual on the GPU", [*base, "--track", "visual"], gpu_dcf=True)

    # E7: a second model on what was found.
    for streams, batch in ((1, 0), (4, 4)):
        base = [V1080, "--model", N, *TRT, "--streams", str(streams), "--batch", str(batch), "--track", "motion"]
        add("E7", f"{streams} stream(s), detector and tracker", base)
        add("E7", f"{streams} stream(s), classifier, each object every 30 pictures",
            [*base, "--classifier", CLS])
        add("E7", f"{streams} stream(s), classifier, each object every picture",
            [*base, "--classifier", CLS, "--reclassify", "1"])

    # E8: drawing and encoding what was found.
    base = [V1080, "--model", N, *TRT]
    add("E8", "detect", base)
    add("E8", "detect, overlay", [*base, "--overlay"])
    add("E8", "detect, overlay, NVENC", [*base, "--overlay", "--encode"])
    add("E8", "decode, NVENC", [V1080, *TRT, "--encode"])
    add("E8", "cpu detect 432p", [V432, "--model", N, *CPU])
    add("E8", "cpu detect, overlay 432p", [V432, "--model", N, *CPU, "--overlay"])

    # E9: everything at once — DeepStream's reference shape.
    for n in (1, 4, 8):
        add(
            "E9",
            f"{n} stream(s) 1080p: interval 2, tracker, classifier, overlay, NVENC",
            [V1080, "--model", N, *TRT, "--streams", str(n), "--batch", str(n) if n > 1 else "0",
             "--interval", "2", "--track", "motion", "--classifier", CLS, "--overlay", "--encode"],
        )
    return m


class GpuSampler:
    """nvidia-smi's view of the GPU, every 200 ms, while a run lasts."""

    FIELDS = "utilization.gpu,utilization.decoder,utilization.encoder,memory.used"

    def __init__(self):
        self.proc = None
        self.samples = []

    def __enter__(self):
        if shutil.which("nvidia-smi"):
            self.proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits", "-lms", "200"],
                stdout=subprocess.PIPE, text=True,
            )
            self.started = time.monotonic()
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            try:
                values = [float(x) for x in line.split(",")]
            except ValueError:
                continue
            # The first second is the pipeline starting.
            if time.monotonic() - self.started > 1.0:
                self.samples.append(values)

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            self.proc.wait()

    def summary(self):
        if not self.samples:
            return {}
        cols = list(zip(*self.samples))
        return {
            "gpu": statistics.mean(cols[0]),
            "dec": statistics.mean(cols[1]),
            "enc": statistics.mean(cols[2]),
            "mem": max(cols[3]),
        }


def run_one(binary, directory, args):
    args = [*args, "--engine-cache-root", ENGINES]
    resolved = [os.path.join(directory, a) if a.endswith((".mp4", ".onnx")) else a for a in args]
    with GpuSampler() as sampler:
        out = subprocess.run([binary, *resolved], capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        sys.stderr.write(f"failed: {' '.join(args)}\n{out.stdout}{out.stderr}\n")
        return None
    result = dict(kv.split("=", 1) for kv in line.split()[1:])
    result.update({k: f"{v:.0f}" for k, v in sampler.summary().items()})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--arm", action="append", required=True, help="NAME=DIRECTORY")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--only")
    p.add_argument("--csv", default="vision-bench.csv")
    a = p.parse_args()
    arms = [tuple(arm.split("=", 1)) for arm in a.arm]

    def binary(directory, gpu_dcf):
        return os.path.join(directory, ("vision_bench-gpu-dcf" if gpu_dcf else "vision_bench") + EXE)

    only = a.only.split(",") if a.only else None
    model_only_bin = os.path.join(arms[0][1], "model_only" + EXE)
    if os.path.exists(model_only_bin) and (not only or "E0" in only):
        model_only(model_only_bin, a.dir, a.reps)
    rows = [r for r in matrix() if not only or r[0] in only]
    rows = [r for r in rows if not r[3] or all(os.path.exists(binary(d, True)) for _, d in arms)]
    # A run of each first, untimed: TensorRT builds an engine for every new
    # model, precision and batch, which takes minutes.
    print(f"{len(rows)} configurations, {len(arms)} arm(s), {a.reps} runs each", file=sys.stderr)
    for exp, label, args, gpu_dcf in rows:
        print(f"  warming {exp} {label}", file=sys.stderr)
        run_one(binary(arms[0][1], gpu_dcf), a.dir, args)

    results = {(i, name): [] for i in range(len(rows)) for name, _ in arms}
    with open(a.csv, "w", newline="") as f:
        writer = None
        for rep in range(a.reps):
            for i, (exp, label, args, gpu_dcf) in enumerate(rows):
                # The arms swap order each time round, so neither always
                # follows the other.
                for name, directory in (arms if rep % 2 == 0 else arms[::-1]):
                    r = run_one(binary(directory, gpu_dcf), a.dir, args)
                    if r is None:
                        continue
                    r.update({"experiment": exp, "configuration": label, "arm": name, "rep": str(rep)})
                    results[(i, name)].append(r)
                    if writer is None:
                        fields = ["experiment", "configuration", "arm", "rep"] + [
                            k for k in r if k not in ("experiment", "configuration", "arm", "rep")
                        ] + ["gpu", "dec", "enc", "mem"]
                        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(fields)), extrasaction="ignore")
                        writer.writeheader()
                    writer.writerow(r)
                    f.flush()
                    print(f"  {rep} {exp} {label} [{name}]: {r['fps']} fps", file=sys.stderr)

    names = [name for name, _ in arms]

    def median(rs, key, digits=0):
        values = [float(r[key]) for r in rs if r.get(key, "-") not in ("-", "")]
        return f"{statistics.median(values):.{digits}f}" if values else "-"

    current = None
    for i, (exp, label, args, _) in enumerate(rows):
        if exp != current:
            current = exp
            print(f"\n### {exp}\n")
            fps_heads = " | ".join(f"{n}: pictures/s" for n in names)
            change = " | change" if len(names) == 2 else ""
            print(f"| configuration | {fps_heads}{change} | CPU cores | GPU % | NVDEC % | NVENC % | GPU MiB | objects/picture |")
            print("|---|" + "---:|" * (len(names) + (1 if len(names) == 2 else 0) + 6))
        per_arm = [results[(i, n)] for n in names]
        cells = []
        for rs in per_arm:
            fps = sorted(float(r["fps"]) for r in rs)
            cells.append(f"{statistics.median(fps):.0f} ({fps[0]:.0f}–{fps[-1]:.0f})" if fps else "failed")
        if len(names) == 2:
            a0, a1 = (statistics.median([float(r["fps"]) for r in rs]) if rs else None for rs in per_arm)
            cells.append(f"{(a1 / a0 - 1) * 100:+.0f}%" if a0 and a1 else "-")
        joined = lambda key, digits=0: " / ".join(median(rs, key, digits) for rs in per_arm)
        objects = next((rs[0]["objects"] for rs in per_arm if rs), "-")
        print(
            f"| {label} | {' | '.join(cells)} | {joined('cpu', 2)} | {joined('gpu')} | "
            f"{joined('dec')} | {joined('enc')} | {joined('mem')} | {objects} |"
        )


if __name__ == "__main__":
    main()
