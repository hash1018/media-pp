// Two NVDEC decoders in one process, through FFmpeg's API alone. A decodes
// throughout; B is opened, decodes on a thread of its own for a while, is
// closed there, and the next B is opened ten seconds later — again and
// again. Both share the CUDA primary context, as media-pp's `CudaDevice`
// does; each decoder's pictures are encoded on h264_nvenc, and each
// decoder is fed 25 pictures a second, as a live stream is.
//
//   nvdec_uaf FILE [CYCLES] [--pin] [--no-primary] [--no-extra] [--no-nvenc] [--b-secs=N]
//
// --pin         hold glibc's mmap threshold at 128 KiB, so that a large
//               block freed is unmapped at once (fence.so does it instead)
// --no-primary  a CUDA context of FFmpeg's own instead of the primary one
// --no-extra    no extra decode surfaces (5 otherwise)
// --no-nvenc    decode only
// --b-secs=N    how long each B decodes before it is closed (5)
//
// See README.md.
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <malloc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// From libavutil/hwcontext_cuda.h, which would need CUDA's cuda.h.
#define AV_CUDA_USE_PRIMARY_CONTEXT (1 << 0)

static int extra = 1, nvenc = 1;
static AVBufferRef *device;

struct dec {
    AVFormatContext *fmt;
    AVCodecContext *ctx;
    AVCodecContext *enc;
    int stream;
    long frames;
    volatile int stop;
};

static enum AVPixelFormat cuda_format(AVCodecContext *ctx, const enum AVPixelFormat *fmts) {
    (void)ctx;
    for (; *fmts != AV_PIX_FMT_NONE; fmts++)
        if (*fmts == AV_PIX_FMT_CUDA) return *fmts;
    fprintf(stderr, "no CUDA format offered\n");
    return AV_PIX_FMT_NONE;
}

static int open_dec(struct dec *d, const char *file) {
    memset(d, 0, sizeof *d);
    if (avformat_open_input(&d->fmt, file, NULL, NULL) < 0) return -1;
    if (avformat_find_stream_info(d->fmt, NULL) < 0) return -1;
    const AVCodec *codec = NULL;
    d->stream = av_find_best_stream(d->fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (d->stream < 0) return -1;
    d->ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(d->ctx, d->fmt->streams[d->stream]->codecpar);
    d->ctx->hw_device_ctx = av_buffer_ref(device);
    d->ctx->get_format = cuda_format;
    if (extra) d->ctx->extra_hw_frames = 5;
    return avcodec_open2(d->ctx, codec, NULL);
}

// h264_nvenc on the decoder's own frames context, opened at the first picture.
static int open_enc(struct dec *d, AVFrame *frame) {
    const AVCodec *codec = avcodec_find_encoder_by_name("h264_nvenc");
    if (!codec) return -1;
    d->enc = avcodec_alloc_context3(codec);
    d->enc->width = frame->width;
    d->enc->height = frame->height;
    d->enc->pix_fmt = AV_PIX_FMT_CUDA;
    d->enc->time_base = (AVRational){1, 25};
    d->enc->framerate = (AVRational){25, 1};
    d->enc->bit_rate = 3628800;
    d->enc->gop_size = 25;
    d->enc->max_b_frames = 0;
    d->enc->hw_frames_ctx = av_buffer_ref(frame->hw_frames_ctx);
    return avcodec_open2(d->enc, codec, NULL);
}

static void close_dec(struct dec *d) {
    avcodec_free_context(&d->enc);
    avcodec_free_context(&d->ctx);
    avformat_close_input(&d->fmt);
}

static void encode(struct dec *d, AVFrame *frame, AVPacket *out) {
    if (!d->enc && open_enc(d, frame) < 0) {
        fprintf(stderr, "cannot open h264_nvenc\n");
        exit(1);
    }
    frame->pts = d->frames;
    frame->pict_type = AV_PICTURE_TYPE_NONE;
    if (avcodec_send_frame(d->enc, frame) < 0) {
        fprintf(stderr, "encode failed\n");
        exit(1);
    }
    while (avcodec_receive_packet(d->enc, out) == 0) av_packet_unref(out);
}

// One packet in, every picture out (and encoded); the file again at its end.
static int step(struct dec *d, AVPacket *pkt, AVFrame *frame, AVPacket *out) {
    int ret = av_read_frame(d->fmt, pkt);
    if (ret == AVERROR_EOF) {
        avformat_seek_file(d->fmt, d->stream, INT64_MIN, 0, 0, 0);
        avcodec_flush_buffers(d->ctx);
        return 0;
    }
    if (ret < 0) return ret;
    if (pkt->stream_index == d->stream) {
        ret = avcodec_send_packet(d->ctx, pkt);
        if (ret < 0 && ret != AVERROR(EAGAIN)) {
            av_packet_unref(pkt);
            return ret;
        }
        while (avcodec_receive_frame(d->ctx, frame) == 0) {
            if (frame->format != AV_PIX_FMT_CUDA) {
                fprintf(stderr, "not decoded on CUDA\n");
                exit(1);
            }
            if (nvenc) encode(d, frame, out);
            d->frames++;
            av_frame_unref(frame);
        }
        // A picture every 40 ms, as a live stream's come.
        struct timespec t = {0, 40 * 1000 * 1000};
        nanosleep(&t, NULL);
    }
    av_packet_unref(pkt);
    return 0;
}

static void *decode_until_stopped(void *arg) {
    struct dec *d = arg;
    AVPacket *pkt = av_packet_alloc(), *out = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    while (!d->stop)
        if (step(d, pkt, frame, out) < 0) {
            fprintf(stderr, "decode failed\n");
            exit(1);
        }
    av_frame_free(&frame);
    av_packet_free(&pkt);
    av_packet_free(&out);
    return NULL;
}

// B's thread: decodes until told to stop, then closes B on this same thread.
static void *decode_then_close(void *arg) {
    decode_until_stopped(arg);
    close_dec(arg);
    return NULL;
}

static void sleep_s(int s) {
    struct timespec t = {s, 0};
    nanosleep(&t, NULL);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s FILE [CYCLES] [--pin] [--no-primary] [--no-extra] [--no-nvenc] [--b-secs=N]\n",
                argv[0]);
        return 2;
    }
    const char *file = argv[1];
    int cycles = 100, pin = 0, primary = 1, b_secs = 5;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--pin")) pin = 1;
        else if (!strcmp(argv[i], "--no-primary")) primary = 0;
        else if (!strcmp(argv[i], "--no-extra")) extra = 0;
        else if (!strcmp(argv[i], "--no-nvenc")) nvenc = 0;
        else if (!strncmp(argv[i], "--b-secs=", 9)) b_secs = atoi(argv[i] + 9);
        else cycles = atoi(argv[i]);
    }
    if (pin && mallopt(M_MMAP_THRESHOLD, 128 * 1024) == 0) {
        fprintf(stderr, "mallopt refused\n");
        return 1;
    }
    av_log_set_level(AV_LOG_ERROR);
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_CUDA, NULL, NULL,
                               primary ? AV_CUDA_USE_PRIMARY_CONTEXT : 0) < 0) {
        fprintf(stderr, "no CUDA device\n");
        return 1;
    }
    printf("pin=%d primary=%d extra=%d nvenc=%d cycles=%d b_secs=%d\n", pin, primary, extra,
           nvenc, cycles, b_secs);

    struct dec a, b;
    if (open_dec(&a, file) < 0) {
        fprintf(stderr, "cannot open A\n");
        return 1;
    }
    pthread_t at, bt;
    pthread_create(&at, NULL, decode_until_stopped, &a);
    for (int n = 1; n <= cycles; n++) {
        if (open_dec(&b, file) < 0) {
            fprintf(stderr, "cannot open B\n");
            return 1;
        }
        pthread_create(&bt, NULL, decode_then_close, &b);
        sleep_s(b_secs);
        b.stop = 1;
        pthread_join(bt, NULL);
        long made = b.frames;
        // The read of the freed block comes within a few seconds of the close.
        sleep_s(10);
        printf("cycle %d: B decoded %ld, A %ld in all\n", n, made, a.frames);
    }
    a.stop = 1;
    pthread_join(at, NULL);
    close_dec(&a);
    av_buffer_unref(&device);
    printf("end\n");
    return 0;
}
