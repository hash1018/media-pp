#!/usr/bin/env python3
"""Makes the speech benchmark's clips from Google's FLEURS test sets, as
docs/benchmarks/speech/method.md describes.

FLEURS reads the same sentences aloud in a hundred languages, one sentence
a recording, each with what was said written down. A clip here is a run of
those recordings back to back, half a second of silence between, in the
order the test set lists them; beside it is a `.txt` of what was said, a
sentence a line, which bench.py scores what was heard against.

Most sentences are read by two speakers, and some recordings are all but
silent — a microphone barely picking the speaker up, 40 dB below the
rest. Of each sentence's recordings the loudest is taken, and a sentence
whose loudest is below QUIET_DB is left out: the benchmark measures the
transcriber, not how a model copes with near-silence, which whisper does
not — it passes over such a sentence whole.

    python3 make_clips.py --fleurs path/to/fleurs --out path/to/clips

--fleurs holds a directory per language, each with that language's
`test.tsv` and `test.tar.gz` from huggingface.co/datasets/google/fleurs
(data/<lang>/test.tsv and data/<lang>/audio/test.tar.gz). The clips are
32-bit float 16 kHz mono WAV, which is what the recordings are: nothing is
re-encoded, so every machine makes the same bytes.
"""

import argparse
import csv
import math
import os
import struct
import sys
import tarfile

# (language directory, clip name, at least how many seconds).
CLIPS = [
    ("ko_kr", "ko-short", 60),
    ("ko_kr", "ko-long", 300),
    ("en_us", "en-short", 60),
    ("en_us", "en-long", 300),
]

RATE = 16_000
# IEEE float, one channel, 32 bits: what every recording is.
FORMAT = (3, 1, RATE, 32)
SAMPLE = 4
GAP = b"\0" * SAMPLE * (RATE // 2)
# A recording quieter than this, as RMS in dB below full scale, is not
# used.
QUIET_DB = -55.0


def read_wav(f, name):
    """The samples of a WAV file in FORMAT, as bytes."""
    data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        sys.exit(f"{name} is not a WAV file")
    at, fmt, samples = 12, None, None
    while at + 8 <= len(data):
        tag, size = data[at : at + 4], struct.unpack("<I", data[at + 4 : at + 8])[0]
        body = data[at + 8 : at + 8 + size]
        if tag == b"fmt ":
            kind, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            fmt = (kind, channels, rate, bits)
        elif tag == b"data":
            samples = body
        at += 8 + size + (size & 1)
    if fmt != FORMAT or samples is None:
        sys.exit(f"{name} is {fmt}, not float 16 kHz mono")
    return samples


def write_wav(path, samples):
    """A WAV file in FORMAT holding `samples`."""
    fmt = struct.pack("<HHIIHH", 3, 1, RATE, RATE * SAMPLE, SAMPLE, 32)
    # A float WAV says in a fact chunk how many samples it holds.
    fact = struct.pack("<I", len(samples) // SAMPLE)
    body = (
        b"WAVE"
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"fact" + struct.pack("<I", len(fact)) + fact
        + b"data" + struct.pack("<I", len(samples)) + samples
    )
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)


def sentences(directory):
    """(sentence, [(file name, what was said)]) for each sentence, in the
    order the test set first lists it, with every recording of it."""
    order, recordings = [], {}
    path = os.path.join(directory, "test.tsv")
    with open(path, encoding="utf-8", newline="") as tsv:
        for row in csv.reader(tsv, delimiter="\t", quoting=csv.QUOTE_NONE):
            sentence_id, name, raw = row[0], row[1], row[2]
            if sentence_id not in recordings:
                order.append(sentence_id)
                recordings[sentence_id] = []
            recordings[sentence_id].append((name, raw))
    for sentence_id in order:
        yield sentence_id, recordings[sentence_id]


def loudness(samples):
    """RMS of float samples, in dB below full scale."""
    values = struct.unpack(f"<{len(samples) // SAMPLE}f", samples)
    rms = math.sqrt(sum(v * v for v in values) / max(len(values), 1))
    return 20 * math.log10(rms) if rms > 0 else -math.inf


def make(directory, out, clip, seconds):
    archive = tarfile.open(os.path.join(directory, "test.tar.gz"))
    members = {os.path.basename(m.name): m for m in archive.getmembers() if m.isfile()}
    frames, said = [], []
    length = 0
    left_out = 0
    for _, recordings in sentences(directory):
        if length >= seconds * RATE:
            break
        heard = [(read_wav(archive.extractfile(members[name]), name), raw) for name, raw in recordings]
        data, raw = max(heard, key=lambda recording: loudness(recording[0]))
        if loudness(data) < QUIET_DB:
            left_out += 1
            continue
        if frames:
            frames.append(GAP)
            length += len(GAP) // SAMPLE
        frames.append(data)
        length += len(data) // SAMPLE
        said.append(raw)
    stem = os.path.join(out, clip)
    write_wav(stem + ".wav", b"".join(frames))
    with open(stem + ".txt", "w", encoding="utf-8", newline="\n") as text:
        text.write("\n".join(said) + "\n")
    print(f"{stem}.wav: {len(said)} sentences, {length / RATE:.1f} s, {left_out} too quiet left out")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fleurs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for language, clip, seconds in CLIPS:
        make(os.path.join(args.fleurs, language), args.out, clip, seconds)


if __name__ == "__main__":
    main()
