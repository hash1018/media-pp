# Cut detection

How well `SwCutDetector` and `CudaCutDetector` find where one shot of an
edited video ends and the next begins, against cuts marked by hand, and
beside ffmpeg's own `scdet` on the same pictures. What a cut is for here:
a tracker starts over on it, so that nothing of one shot is carried into
the next — and a cut missed is worse than one called where there is none,
which costs a tracker no more than a track split in two.

## The films

The first minutes of three Blender Foundation films, CC BY, fetched as they
are published — no file of them is in this repository:

| Film | Minutes | Size | What is in it |
|---|---:|---|---|
| Tears of Steel | 7 | 1280×534, 24/s | Live action, dialogue and effects shots |
| Sintel | 5 | 1920×818, 24/s | Animated, snow and fighting, then a dark tent and a town |
| Elephants Dream | 5 | 1024×576, 24/s | Animated, dark, wires and machines, fades through black |

```sh
ffmpeg -i https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov -t 420 -map 0:v:0 -c copy tos.mp4
ffmpeg -i https://download.blender.org/demo/movies/Sintel.2010.1080p.mkv -t 300 -map 0:v:0 -c copy sintel.mkv
ffmpeg -i https://download.blender.org/ED/ED_1024.avi -t 300 -map 0:v:0 -c:v h264_nvenc -b:v 6M ed.mp4
```

Elephants Dream is published as MPEG-4 Part 2 in AVI and was re-encoded to
H.264 for NVDEC; the marks were made on the re-encoding, and are picture
numbers, which another encoding of the same minutes keeps.

## The marks

Every picture either the detector's score or `scdet` took for a possible
cut, at thresholds well below either's — 360 of them in the 19 minutes —
was looked at beside the pictures around it and marked a cut or not. A cut
the two together missed is not in the marks, so recall is of what either
could have found. They are in [cuts/](cuts/), one picture a line:

- `C` — a cut: the first picture of the new shot. Found within a picture
  of it counts.
- `F` — the start of a fade back from black. Found anywhere in the next
  second counts: where a fade begins a shot is a matter of taste.
- `B` — a cut into black. Not to be found, since a shot begins where the
  picture comes back, and not held against a detector that finds it.

222 cuts are to be found: Tears of Steel 92, Sintel 72, Elephants Dream 58.

## Results

`cut_detect` with the default options, on Windows, RTX 3050:

```sh
cargo run --release -p cut_detect -- tos.mp4 --cuda > tos.out
python docs/benchmarks/vision/cuts/score.py docs/benchmarks/vision/cuts/tears-of-steel.txt tos.out ...
```

| | Found | False | Missed | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Tears of Steel | 92 | 5 | 0 | | |
| Sintel | 71 | 2 | 1 | | |
| Elephants Dream | 57 | 4 | 1 | | |
| **All** | **220** | **11** | **2** | **0.952** | **0.991** |
| ffmpeg `scdet`, threshold 8 — its best | 191 | 18 | 31 | 0.914 | 0.860 |
| ffmpeg `scdet`, threshold 3 — to miss as few | 215 | 151 | 7 | 0.587 | 0.968 |

`SwCutDetector` finds the same cuts, picture for picture, in all three: the
two make the same thumbnail of each picture.

The two missed: a cut in Sintel's fight between two pictures already a
long way apart, at twice the average of those before it where 2.5 is
asked for, and short of the 40 that lets twice do; and one in Elephants
Dream between two dark, alike shots, six out of 510 apart. The false ones
are mostly a picture flooded by light — a rocket's exhaust, a hologram
switched on — and the start of a fast move in animation.

The options were chosen on these marks: there is no other set to check
them on yet, so the figures are what the options were fitted to, not what
another video will show. Of what was tried —

- each pictures' distance against the average of the eight before it, at
  three times it: precision 0.941, recall 0.937. A cut in a fight, where
  every picture is far from the last, stood too little above them, and
  the picture of blur running up to a cut was taken for it, two pictures
  early;
- the cut put on the furthest picture where the next one or two are
  further still, and a picture past 40 a cut at twice the average: both
  kinds of miss went — precision 0.952, recall 0.973 at three times, and
  0.940 and 0.991 at 2.5, three more false ones in 19 minutes for four
  more cuts found. 2.5 is the default.
- a picture far from the last but alike it in shape, its luma cells
  correlated with the last's at 0.85 or more, taken for a light over the
  same shot rather than a cut. A concert recording on a stage of
  swinging, switching lights had 137 cuts, 65 of them one shot brightened
  or dimmed where the singer stood and not behind her — each such pair
  correlated over 0.85, the cut that followed at about 0. Here it lost no
  cut and three false ones: precision 0.952, recall 0.991. At 0.8 a cut in
  Sintel went too; at 0.9 precision was 0.936, a false one more than
  without it. 0.85 is what the detectors do.

## Speed

Decoding included, one stream:

| | Tears of Steel, 1280×534 | Sintel, 1920×818 | Elephants Dream, 1024×576 |
|---|---:|---:|---:|
| CPU (`SwDecoder`, `SwCutDetector`) | 473/s | 132/s | 321/s |
| CUDA (`CudaDecoder`, `CudaCutDetector`) | 1589/s | 791/s | 1803/s |

Each picture's thumbnail on the GPU is two launches of the kernel the CUDA
overlay cuts a mosaic with, and under 7,000 floats copied down.
