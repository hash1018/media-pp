#!/bin/sh
# Builds the FFmpeg media-pp is built against, and says how to point the
# build at it — the Linux and macOS counterpart of setup-ffmpeg.ps1.
#
# media-pp needs FFmpeg 8.0's development files: its bindings break on 8.1's
# headers, and a distribution's or Homebrew's FFmpeg is whichever release
# they ship. This builds 8.0.1 from vcpkg at the tag CI uses (the same one
# .github/actions/setup-ffmpeg pins), into a folder of your own, and writes
# an `env.sh` there that sets:
#
#   FFMPEG_DIR       where the build finds the headers; ffmpeg-sys-next reads
#                    it before looking anywhere else
#   LD_LIBRARY_PATH  (Linux) or DYLD_LIBRARY_PATH (macOS): where programs
#                    built against it find its shared libraries
#
# Building FFmpeg takes tens of minutes the first time.
#
#   scripts/setup-ffmpeg.sh [--into DIR] [--vulkan] [--features LIST]
#   . ~/.local/share/media-pp/ffmpeg/env.sh
#
# By default the build carries OpenH264, and on Linux nvcodec too — NVIDIA's
# decode and encode, loaded at run time, so harmless without an NVIDIA GPU —
# with FFmpeg's CUDA filters (scale_cuda) compiled by clang, which media-pp's
# CUDA scaling and compositing use. --vulkan adds what the crate's `vulkan`
# feature builds against.
set -eu

# Keep in step with .github/actions/setup-ffmpeg/action.yml.
VCPKG_REF=2026.01.16
PORT_VERSION=8.0.1

into="${XDG_DATA_HOME:-$HOME/.local/share}/media-pp/ffmpeg"
vulkan=no
features=
while [ $# -gt 0 ]; do
    case "$1" in
        --into) into="$2"; shift 2 ;;
        --vulkan) vulkan=yes; shift ;;
        --features) features="$2"; shift 2 ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

case "$(uname -s)" in
    Linux)
        triplet=x64-linux-dynamic
        libvar=LD_LIBRARY_PATH
        default=openh264,nvcodec
        need="git curl zip unzip tar pkg-config make nasm autoconf automake libtool python3 clang"
        hint="on Debian or Ubuntu: sudo apt install build-essential git curl zip unzip tar pkg-config nasm autoconf automake libtool python3 clang"
        ;;
    Darwin)
        triplet=arm64-osx-dynamic
        libvar=DYLD_LIBRARY_PATH
        default=openh264
        need="git curl zip unzip tar pkg-config"
        hint="xcode-select --install, then: brew install pkgconf"
        ;;
    *) echo "this script knows Linux and macOS; on Windows use setup-ffmpeg.ps1" >&2; exit 1 ;;
esac
[ "$vulkan" = yes ] && default="$default,vulkan"
features="${features:-$default}"

missing=
for tool in $need; do
    command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
if [ -n "$missing" ]; then
    echo "missing:$missing" >&2
    echo "$hint" >&2
    exit 1
fi

vcpkg="$into/vcpkg"
if [ ! -d "$vcpkg/.git" ]; then
    echo "Fetching vcpkg $VCPKG_REF into $vcpkg"
    mkdir -p "$into"
    git clone --quiet --depth 1 --branch "$VCPKG_REF" https://github.com/microsoft/vcpkg.git "$vcpkg"
fi

# A vcpkg already there may be at another tag, whose ffmpeg port builds
# another FFmpeg: refused rather than built from.
if ! grep -q "\"version\": \"$PORT_VERSION\"" "$vcpkg/ports/ffmpeg/vcpkg.json"; then
    echo "the vcpkg at $vcpkg does not build FFmpeg $PORT_VERSION: remove it, or pass another --into" >&2
    exit 1
fi

# The port configures with --disable-autodetect and enables no CUDA kernel
# compiler, so its nvcodec build has NVENC and NVDEC but no scale_cuda, which
# media-pp's CUDA scaler and compositor are built on. clang compiles them to
# PTX, keeping the licence LGPL (nvcc would make it nonfree).
case ",$features," in
    *,nvcodec,*)
        if [ "$triplet" = x64-linux-dynamic ]; then
            port="$vcpkg/ports/ffmpeg/portfile.cmake"
            grep -q -- '--enable-cuda-llvm' "$port" ||
                sed -i 's/--enable-ffnvcodec")/--enable-ffnvcodec --enable-cuda-llvm")/' "$port"
            grep -q -- '--enable-ffnvcodec --enable-cuda-llvm")' "$port" || {
                echo "the ffmpeg port's nvcodec options are not where this script expects them" >&2
                exit 1
            }
        fi
        ;;
esac

[ -x "$vcpkg/vcpkg" ] || "$vcpkg/bootstrap-vcpkg.sh" -disableMetrics
echo "Building ffmpeg[$features] — this takes a while the first time."
"$vcpkg/vcpkg" install "ffmpeg[$features]:$triplet"
# Nothing links the debug variant, and it is half the tree.
rm -rf "$vcpkg/installed/$triplet/debug"

prefix="$vcpkg/installed/$triplet"
version="$prefix/include/libavcodec/version_major.h"
[ -f "$version" ] || { echo "no FFmpeg headers under $prefix" >&2; exit 1; }
grep -Eq '#define[[:space:]]+LIBAVCODEC_VERSION_MAJOR[[:space:]]+62\b' "$version" ||
    { echo "the headers under $prefix are not FFmpeg 8.0's" >&2; exit 1; }

cat > "$into/env.sh" <<EOF
# FFmpeg $PORT_VERSION for media-pp, from scripts/setup-ffmpeg.sh.
export FFMPEG_DIR="$prefix"
export $libvar="$prefix/lib\${$libvar:+:\$$libvar}"
export PKG_CONFIG_PATH="$prefix/lib/pkgconfig\${PKG_CONFIG_PATH:+:\$PKG_CONFIG_PATH}"
EOF
echo "FFmpeg $PORT_VERSION is ready. In each shell that builds or runs media-pp:"
echo "  . \"$into/env.sh\""
echo "or add that line to your shell's profile. Then check the machine with: cargo run -p doctor"
