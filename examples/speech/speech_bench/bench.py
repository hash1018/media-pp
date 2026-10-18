#!/usr/bin/env python3
"""Runs speech_bench over a matrix of configurations, as
docs/benchmarks/speech describes, scores what each heard against what was
said, and prints a Markdown table for each experiment.

Every configuration runs --reps times, the configurations taking turns
(a, b, c, a, b, c, ...) rather than one after another, so that a GPU
warming up or something else starting on the machine lands on all of them
alike; the table gives the median speed and the spread. Where nvidia-smi
is found, the GPU's utilisation and the memory the run added are sampled
every 200 ms.

    python3 bench.py --dir path/to/clips-and-models --bin path/to/builds \
        [--backends cpu,cuda,vulkan] [--reps 3] [--only S0,S2] [--csv results.csv]

--bin holds speech_bench and model_only, built with the GPU backends to
compare (`--features cuda,vulkan`, or `metal` on a Mac). --dir holds the
clips make_clips.py makes — ko-short, ko-long, en-short, en-long, each .wav
with its .txt — and the whisper.cpp models the matrix names. --backends
says which of the build's backends to run, the first GPU among them being
the one the single-backend experiments use.

Accuracy is the character error rate (CER) for Korean, whose spacing is
loose enough that words are a poor unit, and the word error rate (WER)
for English: the edits that turn what was heard into what was said, over
what was said. Both sides are compared as letters and digits alone,
lower-cased and NFKC-normalised, so punctuation and capitals cost nothing.
"""

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata

EXE = ".exe" if os.name == "nt" else ""
MACOS = sys.platform == "darwin"

KO_SHORT, KO_LONG, EN_SHORT, EN_LONG = "ko-short.wav", "ko-long.wav", "en-short.wav", "en-long.wav"
TINY, BASE, SMALL, MEDIUM = "ggml-tiny.bin", "ggml-base.bin", "ggml-small.bin", "ggml-medium.bin"
TURBO, TURBO_Q5, LARGE = "ggml-large-v3-turbo.bin", "ggml-large-v3-turbo-q5_0.bin", "ggml-large-v3.bin"
MODELS = [TINY, BASE, SMALL, MEDIUM, TURBO, TURBO_Q5, LARGE]
# What the CPU hears in tolerable time: large-v3-turbo ran at under a
# tenth of real time there.
CPU_MODELS = [TINY, BASE, SMALL]


def language(clip):
    return clip.split("-")[0]


def matrix(backends):
    """(experiment, label, binary, arguments)."""
    gpus = [b for b in backends if b != "cpu"]
    gpu = gpus[0] if gpus else None
    cpu = "cpu" in backends
    m = []

    def add(exp, label, args, binary="speech_bench"):
        m.append((exp, label, binary, args))

    def given(clip):
        return ["--language", language(clip)]

    # S0: the model alone, the whole clip in one call.
    if gpu:
        for model in MODELS:
            add("S0", f"{gpu} {model}", [KO_LONG, "--model", model, "--backend", gpu, *given(KO_LONG)], "model_only")
    if cpu:
        for model in CPU_MODELS:
            add("S0", f"cpu {model} (short clip)", [KO_SHORT, "--model", model, "--backend", "cpu", *given(KO_SHORT)], "model_only")

    # S1: who runs the model.
    for model in CPU_MODELS + [TURBO]:
        for backend in backends:
            if backend == "cpu" and model not in CPU_MODELS:
                continue
            add("S1", f"{backend} {model}", [KO_SHORT, "--model", model, "--backend", backend, *given(KO_SHORT)])

    # S2: the model's size and precision, on every GPU.
    for model in MODELS:
        for backend in gpus:
            add("S2", f"{backend} {model}", [KO_LONG, "--model", model, "--backend", backend, *given(KO_LONG)])

    # S3: the same in English.
    if gpu:
        for model in MODELS:
            add("S3", f"{gpu} {model}", [EN_LONG, "--model", model, "--backend", gpu, *given(EN_LONG)])

    if gpu:
        # S4: the language detected or given, and words timed by estimate
        # or against the audio.
        base = [KO_LONG, "--model", TURBO, "--backend", gpu]
        add("S4", "language detected", base)
        add("S4", "language given", [*base, *given(KO_LONG)])
        add("S4", "language given, words aligned", [*base, *given(KO_LONG), "--align"])

        # S5: how much new sound each round hears.
        for chunk in (2000, 4000, 8000):
            add("S5", f"chunk {chunk} ms", [*base, *given(KO_LONG), "--chunk-ms", str(chunk)])

        # S6: live, the sound arriving as it is said.
        for model in (BASE, SMALL, TURBO, LARGE):
            add("S6", f"{gpu} {model}", [KO_SHORT, "--model", model, "--backend", gpu, *given(KO_SHORT), "--realtime"])
        for chunk in (2000, 8000):
            add("S6", f"{gpu} {TURBO}, chunk {chunk} ms",
                [KO_SHORT, "--model", TURBO, "--backend", gpu, *given(KO_SHORT), "--realtime", "--chunk-ms", str(chunk)])
    if cpu:
        add("S6", f"cpu {BASE}", [KO_SHORT, "--model", BASE, "--backend", "cpu", *given(KO_SHORT), "--realtime"])
    return m


def normalized(text):
    """Letters and digits alone, lower-cased, everything else a space."""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(c if unicodedata.category(c)[0] in "LN" else " " for c in text)


def distance(a, b):
    """Levenshtein distance between two sequences."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def score(said, heard):
    """(CER %, WER %) of what was heard against what was said."""
    said, heard = normalized(said), normalized(heard)
    chars = lambda t: [c for c in t if not c.isspace()]
    cer = distance(chars(said), chars(heard)) / max(len(chars(said)), 1) * 100
    wer = distance(said.split(), heard.split()) / max(len(said.split()), 1) * 100
    return cer, wer


class GpuSampler:
    """nvidia-smi's view of the GPU, every 200 ms, while a run lasts."""

    FIELDS = "utilization.gpu,memory.used"

    def __init__(self):
        self.proc = None
        self.samples = []
        self.idle = None

    def __enter__(self):
        if shutil.which("nvidia-smi"):
            query = ["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits"]
            first = subprocess.run(query, capture_output=True, text=True).stdout.splitlines()
            if first:
                self.idle = float(first[0].split(",")[1])
            self.proc = subprocess.Popen([*query, "-lms", "200"], stdout=subprocess.PIPE, text=True)
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
            # The first second is the model loading.
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
        added = max(cols[1]) - self.idle if self.idle is not None else max(cols[1])
        return {"gpu": statistics.mean(cols[0]), "mem": added}


def run_one(bindir, binary, directory, args, scratch):
    resolved = [os.path.join(directory, a) if a.endswith((".wav", ".bin")) else a for a in args]
    heard_path = os.path.join(scratch, "heard.txt")
    if os.path.exists(heard_path):
        os.remove(heard_path)
    with GpuSampler() as sampler:
        out = subprocess.run(
            [os.path.join(bindir, binary + EXE), *resolved, "--transcript", heard_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=scratch,
        )
    line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        sys.stderr.write(f"failed: {binary} {' '.join(args)}\n{out.stdout[-2000:]}{out.stderr[-2000:]}\n")
        return None
    result = dict(kv.split("=", 1) for kv in line.split()[1:])
    result.update({k: f"{v:.0f}" for k, v in sampler.summary().items()})
    clip = os.path.join(directory, args[0])
    with open(clip[:-4] + ".txt", encoding="utf-8") as f:
        said = f.read()
    with open(heard_path, encoding="utf-8") as f:
        heard = f.read()
    cer, wer = score(said, heard)
    result.update({"cer": f"{cer:.1f}", "wer": f"{wer:.1f}"})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--bin", required=True)
    p.add_argument("--backends", default="cpu,metal" if MACOS else "cpu,cuda,vulkan")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--only")
    p.add_argument("--csv", default="speech-bench.csv")
    a = p.parse_args()
    # Each run starts in a directory of its own, for its logs.
    a.dir, a.bin = os.path.abspath(a.dir), os.path.abspath(a.bin)

    only = a.only.split(",") if a.only else None
    rows = [r for r in matrix(a.backends.split(",")) if not only or r[0] in only]
    scratch = tempfile.mkdtemp(prefix="speech-bench-")
    # A run of each first, untimed: a GPU backend's first run compiles its
    # shaders or kernels for the card, and the model files come into the
    # page cache.
    print(f"{len(rows)} configurations, {a.reps} runs each", file=sys.stderr)
    for exp, label, binary, args in rows:
        print(f"  warming {exp} {label}", file=sys.stderr)
        run_one(a.bin, binary, a.dir, args, scratch)

    results = {i: [] for i in range(len(rows))}
    with open(a.csv, "w", newline="", encoding="utf-8") as f:
        writer = None
        for rep in range(a.reps):
            for i, (exp, label, binary, args) in enumerate(rows):
                r = run_one(a.bin, binary, a.dir, args, scratch)
                if r is None:
                    continue
                r.update({"experiment": exp, "configuration": label, "rep": str(rep)})
                results[i].append(r)
                if writer is None:
                    head = ["experiment", "configuration", "rep"]
                    fields = head + [k for k in r if k not in head] + ["gpu", "mem", "lag", "lag_p90", "lag_max"]
                    writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(fields)), extrasaction="ignore")
                    writer.writeheader()
                writer.writerow(r)
                f.flush()
                print(f"  {rep} {exp} {label}: {r['speed']}x, CER {r['cer']}%, WER {r['wer']}%", file=sys.stderr)

    def median(rs, key, digits=0):
        values = [float(r[key]) for r in rs if r.get(key, "-") not in ("-", "")]
        return f"{statistics.median(values):.{digits}f}" if values else "-"

    current = None
    for i, (exp, label, binary, args) in enumerate(rows):
        rs = results[i]
        live = "--realtime" in args
        if exp != current:
            current = exp
            print(f"\n### {exp}\n")
            if live:
                print("| configuration | lag, s | lag p90, s | lag max, s | CER % | CPU cores | GPU % | GPU MiB |")
                print("|---|---:|---:|---:|---:|---:|---:|---:|")
            else:
                print("| configuration | × real time | CER % | WER % | CPU cores | GPU % | GPU MiB | load, s |")
                print("|---|---:|---:|---:|---:|---:|---:|---:|")
        if not rs:
            print(f"| {label} | failed |")
            continue
        if live:
            print(f"| {label} | {median(rs, 'lag', 2)} | {median(rs, 'lag_p90', 2)} | {median(rs, 'lag_max', 2)} | "
                  f"{median(rs, 'cer', 1)} | {median(rs, 'cpu', 2)} | {median(rs, 'gpu')} | {median(rs, 'mem')} |")
        else:
            speed = sorted(float(r["speed"]) for r in rs)
            spread = f"{statistics.median(speed):.1f} ({speed[0]:.1f}–{speed[-1]:.1f})"
            print(f"| {label} | {spread} | {median(rs, 'cer', 1)} | {median(rs, 'wer', 1)} | {median(rs, 'cpu', 2)} | "
                  f"{median(rs, 'gpu')} | {median(rs, 'mem')} | {median(rs, 'ready', 2)} |")


if __name__ == "__main__":
    main()
