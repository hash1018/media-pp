//! Link contracts (see `crate::contract`): what is refused before anything
//! runs, and what must still link.

use super::*;

// ---------------------------------------------------------------------
// Link contracts (see `crate::contract`).
//
// These cover the check itself rather than any one element: that a
// mismatch is refused before anything runs, that a `Queue` in the middle
// does not hide one, that an undeclared contract still links, and that
// the pad-to-branch boundary an attach crosses is checked too.
// ---------------------------------------------------------------------

/// A terminal sink declaring whatever a given test needs to check against.
struct DeclaringSink {
    name: Arc<str>,
    pp_log: PpLog,
    contract: InputContract,
}

impl DeclaringSink {
    fn boxed(name: &'static str, contract: InputContract) -> Box<Self> {
        Box::new(Self {
            name: name.into(),
            pp_log: element_pp_log(ElementType::Other, name, None),
            contract,
        })
    }
}

impl Element for DeclaringSink {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Other
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for DeclaringSink {
    fn input_contract(&self) -> InputContract {
        self.contract
    }

    fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

fn contract_context() -> Arc<Context> {
    let (bus, _rx) = Bus::new();
    let graph = PipelineGraph::new();
    let source_id = graph.add_source(ElementType::Other, "source".into());
    Arc::new(Context::for_test(bus, "contracts", graph, source_id))
}

/// A decoder built without any fixture — only `codec_type`/`codec_id`
/// decide which one `SwDecoder::new` opens. Same approach as that
/// element's own tests, so these run without waiting on a fixture to be
/// built.
fn decoder(name: &str, medium: ffmpeg::media::Type, codec: ffmpeg::codec::Id) -> SwDecoder {
    let mut params = ffmpeg::codec::Parameters::new();
    // SAFETY: `as_mut_ptr` on parameters this test just created and still
    // owns exclusively; both are plain fields of `AVCodecParameters`.
    unsafe {
        (*params.as_mut_ptr()).codec_type = medium.into();
        (*params.as_mut_ptr()).codec_id = codec.into();
    }
    SwDecoder::new(name, params).expect("the built-in decoder must open")
}

fn video_decoder(name: &str) -> SwDecoder {
    decoder(name, ffmpeg::media::Type::Video, ffmpeg::codec::Id::H264)
}

fn audio_decoder(name: &str) -> SwDecoder {
    decoder(name, ffmpeg::media::Type::Audio, ffmpeg::codec::Id::AAC)
}

fn video_frames() -> InputContract {
    InputContract::Fixed(PortContract::frame(
        MediaKind::VideoFrame,
        MemoryDomain::System,
    ))
}

#[test]
fn a_decoder_reaches_a_matching_sink_through_a_queue() {
    contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .queue("q", 4)
        .to(DeclaringSink::boxed("renderer", video_frames()))
        .expect("decoded video into a video sink is a valid link");
}

/// The one that makes the whole feature worth having: a `Queue` is a
/// thread boundary, not a transform, so a mismatch two stages apart must
/// still be caught — and must still name the decoder that actually
/// produces the frames rather than the queue that would have relayed them.
#[test]
fn a_queue_in_the_middle_does_not_hide_a_mismatch() {
    let Err(error) = contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .queue("q", 4)
        .to(DeclaringSink::boxed(
            "muxer",
            InputContract::Fixed(PortContract::packet(MediaKind::VideoPacket)),
        ))
    else {
        panic!("decoded frames cannot be fed to a packet-only sink");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(&*producer, "decoder");
    assert_eq!(&*consumer, "muxer");
}

/// Two `SwDecoder`s of the same `ElementType` legitimately produce
/// different kinds, decided by the stream parameters each was built with.
/// This is what makes a contract a property of the instance rather than
/// something a table keyed by `ElementType` could ever answer.
#[test]
fn an_audio_decoder_is_rejected_where_a_video_decoder_would_link() {
    contract_context()
        .branch()
        .pipe(video_decoder("video"))
        .to(DeclaringSink::boxed("encoder", video_frames()))
        .expect("the video decoder is the case this sink accepts");

    let Err(error) = contract_context()
        .branch()
        .pipe(audio_decoder("audio"))
        .to(DeclaringSink::boxed("encoder", video_frames()))
    else {
        panic!("an audio decoder cannot feed a video-only sink");
    };

    assert!(
        matches!(
            error,
            crate::Error::GraphError(GraphError::IncompatibleLink { .. })
        ),
        "got {error}"
    );
}

/// Over-rejection is the real risk of a check like this: a wrong
/// declaration refuses a pipeline that works. An element that declares
/// nothing must keep linking to anything, exactly as before contracts
/// existed.
#[test]
fn an_undeclared_contract_still_links_to_anything() {
    contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .to(Box::new(NoOpSink {
            name: "undeclared".into(),
            pp_log: element_pp_log(ElementType::Other, "undeclared", None),
        }))
        .expect("an Unknown contract is missing information, not a refusal");
}

/// The boundary `ChainBuilder::to` cannot see: a branch is built on its
/// own and only meets the pad feeding it at attach time. A refusal there
/// has to leave both the pad and the graph exactly as they were.
#[test]
fn attaching_a_packet_pad_to_a_video_branch_changes_nothing() {
    let context = contract_context();
    let branch = context
        .branch()
        .queue("q", 4)
        .to(DeclaringSink::boxed("encoder", video_frames()))
        .expect("the branch itself is consistent");
    let before = context.graph.snapshot();

    let mut pad = SrcPad::with_contract(
        "demuxer_src",
        OutputContract::Fixed(PortContract::packet(MediaKind::VideoPacket)),
    );
    let error = context
        .attach_pad(&mut pad, branch)
        .expect_err("a packet pad cannot feed a branch that decodes nothing");

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(&*producer, "demuxer_src");
    assert_eq!(
        &*consumer, "encoder",
        "the leading Queue defers the question rather than answering it"
    );

    assert!(!pad.is_linked(), "a refused attach must not link the pad");
    let after = context.graph.snapshot();
    assert_eq!(before.revision, after.revision);
    assert_eq!(before.nodes.len(), after.nodes.len());
}

/// The declaration on a real source pad rather than on a test double.
/// Needs a container to open, so it only runs where one is configured.
#[test]
fn a_demuxer_declares_packets_on_every_stream_pad() {
    let Some(path) = try_test_video() else { return };
    let (mut demuxer, streams) =
        FileDemuxer::open("demuxer", &path).expect("the fixture must open");

    // Per stream, from the medium the container announced — a video pad
    // and an audio pad are both `MediaBuffer::Packet` but not the same
    // contract, which is the whole point of splitting the kind.
    for (pad, stream) in demuxer.src_pads().iter().zip(&streams) {
        let expected = match MediaKind::packet_for(stream.kind) {
            Some(kind) => OutputContract::Fixed(PortContract::packet(kind)),
            None => OutputContract::Unknown,
        };
        assert_eq!(pad.contract(), expected);
    }
}

/// What the memory domain was added for. Both frames here are
/// `MediaBuffer::Video`, so nothing about the buffer type distinguishes
/// them — only the domain says one lives in system memory and the other
/// in a texture, and only that catches the missing `D3d11Upload`.
#[cfg(all(target_os = "windows", feature = "d3d11"))]
#[test]
fn a_system_memory_frame_cannot_feed_a_d3d11_filter() {
    use crate::elements::{D3d11ScalerFormat, D3d11Upload};

    let Some((device, d3d_context)) = crate::test_support::try_d3d11_device() else {
        eprintln!("skipped: no D3D11 hardware device available");
        return;
    };
    let gpu_frames = InputContract::Fixed(PortContract::frame(
        MediaKind::VideoFrame,
        MemoryDomain::D3d11,
    ));
    // A device existing is not the same as it supporting the video
    // processor this scaler opens: CI machines routinely have one without
    // the other, and that is a missing capability to skip on, not a
    // failure of what this test is checking.
    let scaler = |name: &str| {
        crate::elements::D3d11Scaler::new(
            name,
            &device,
            d3d_context.clone(),
            D3d11ScalerFormat::Preserve,
            64,
            64,
        )
    };
    if let Err(error) = scaler("probe") {
        eprintln!("skipped: this D3D11 device has no usable video processor ({error})");
        return;
    }
    let scaler = |name: &str| scaler(name).expect("the probe above already opened one");

    let Err(error) = contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .pipe(scaler("scaler"))
        .to(DeclaringSink::boxed("renderer", gpu_frames))
    else {
        panic!("a software decoder's frames never reach a GPU scaler");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(&*producer, "decoder");
    assert_eq!(&*consumer, "scaler");

    // The same chain with the upload that was missing.
    contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .pipe(D3d11Upload::new("upload", &device, 64, 64))
        .pipe(scaler("scaler"))
        .to(DeclaringSink::boxed("renderer", gpu_frames))
        .expect("a D3d11Upload is exactly what makes this chain valid");
}

/// The reverse direction, which is just as easy to get wrong: a device
/// texture handed to a CPU filter.
#[cfg(all(target_os = "windows", feature = "d3d11"))]
#[test]
fn a_d3d11_frame_cannot_feed_a_cpu_filter() {
    use crate::elements::{D3d11Upload, SwScaler};

    let Some((device, _d3d_context)) = crate::test_support::try_d3d11_device() else {
        eprintln!("skipped: no D3D11 hardware device available");
        return;
    };

    let Err(error) = contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .pipe(D3d11Upload::new("upload", &device, 64, 64))
        .pipe(SwScaler::new(
            "scaler",
            ffmpeg::format::Pixel::YUV420P,
            64,
            64,
            ffmpeg::software::scaling::Flags::BILINEAR,
        ))
        .to(DeclaringSink::boxed("sink", video_frames()))
    else {
        panic!("a device texture has no CPU-readable planes for swscale");
    };

    assert!(
        matches!(
            error,
            crate::Error::GraphError(GraphError::IncompatibleLink { .. })
        ),
        "got {error}"
    );
}

/// A port that deals in more than one kind is why the contract holds a
/// set rather than a single `MediaKind`: `FrameCounter` tallies decoded
/// buffers of either medium, and both must link.
#[test]
fn a_frame_counter_takes_either_decoded_medium() {
    use crate::elements::FrameCounter;

    for decoder in [video_decoder("video"), audio_decoder("audio")] {
        let (counter, _count) = FrameCounter::new("counter");
        contract_context()
            .branch()
            .pipe(decoder)
            .to(Box::new(counter))
            .expect("a decoded-buffer counter accepts both video and audio");
    }
}

/// The same counter pair in the other direction: `PacketCounter` deals in
/// encoded data only, so the decoder that feeds its sibling cannot feed it.
#[test]
fn a_decoded_frame_cannot_feed_a_packet_only_sink() {
    use crate::elements::PacketCounter;

    let (counter, _count) = PacketCounter::new("counter");
    let Err(error) = contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .to(Box::new(counter))
    else {
        panic!("decoded frames are not packets");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(&*producer, "decoder");
    assert_eq!(&*consumer, "counter");
}

/// Video and audio are both decoded frames, so nothing but the media kind
/// separates a video filter from an audio one.
#[test]
fn an_audio_filter_refuses_video_frames() {
    use crate::elements::AudioVolume;

    let (volume, _handle) = AudioVolume::new("volume");
    let Err(error) = contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .pipe(volume)
        .to(DeclaringSink::boxed("sink", video_frames()))
    else {
        panic!("a gain filter has no samples to scale in a video frame");
    };

    assert!(
        matches!(
            error,
            crate::Error::GraphError(GraphError::IncompatibleLink { .. })
        ),
        "got {error}"
    );

    // The audio decoder is the case that filter is for.
    let (volume, _handle) = AudioVolume::new("volume");
    contract_context()
        .branch()
        .pipe(audio_decoder("decoder"))
        .pipe(volume)
        .to(DeclaringSink::boxed(
            "sink",
            InputContract::Fixed(PortContract::frame(
                MediaKind::AudioFrame,
                MemoryDomain::System,
            )),
        ))
        .expect("decoded audio through a gain filter is the intended chain");
}

/// The filters that work on the signal itself declare the same as the gain
/// one: decoded audio in system memory, refused where it is video and taken
/// where it is audio.
#[test]
fn the_signal_filters_refuse_video_frames_and_take_audio() {
    use crate::element::Filter;

    type Make = fn() -> Box<dyn Filter>;
    let filters: Vec<(&str, Make)> = vec![
        ("gate", || {
            Box::new(crate::elements::AudioGate::new("gate").0)
        }),
        ("compressor", || {
            Box::new(crate::elements::AudioCompressor::new("compressor").0)
        }),
        ("limiter", || {
            Box::new(crate::elements::AudioLimiter::new("limiter").0)
        }),
        #[cfg(feature = "rnnoise")]
        ("denoise", || {
            Box::new(crate::elements::NoiseSuppressor::new("denoise"))
        }),
    ];
    for (name, make) in filters {
        let Err(error) = contract_context()
            .branch()
            .pipe(video_decoder("decoder"))
            .pipe(make())
            .to(DeclaringSink::boxed("sink", video_frames()))
        else {
            panic!("{name} took video frames");
        };
        assert!(
            matches!(
                error,
                crate::Error::GraphError(GraphError::IncompatibleLink { .. })
            ),
            "{name}: got {error}"
        );

        contract_context()
            .branch()
            .pipe(audio_decoder("decoder"))
            .pipe(make())
            .to(DeclaringSink::boxed(
                "sink",
                InputContract::Fixed(PortContract::frame(
                    MediaKind::AudioFrame,
                    MemoryDomain::System,
                )),
            ))
            .unwrap_or_else(|error| panic!("{name} refused decoded audio: {error}"));
    }
}

/// A video effect declares decoded video in system memory — the mirror of
/// the audio filters above: refused behind an audio decoder, taken behind a
/// video one.
#[test]
fn a_video_effect_takes_video_frames_and_refuses_audio() {
    use crate::elements::{ColorCorrection, SwVideoEffect, VideoEffect};

    let effect = || {
        SwVideoEffect::new(
            "effect",
            VideoEffect::ColorCorrection(ColorCorrection::default()),
        )
        .0
    };
    let Err(error) = contract_context()
        .branch()
        .pipe(audio_decoder("decoder"))
        .pipe(effect())
        .to(DeclaringSink::boxed("sink", video_frames()))
    else {
        panic!("a video effect has no pixels to change in an audio frame");
    };
    assert!(
        matches!(
            error,
            crate::Error::GraphError(GraphError::IncompatibleLink { .. })
        ),
        "got {error}"
    );

    contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .pipe(effect())
        .to(DeclaringSink::boxed("sink", video_frames()))
        .expect("decoded video through a video effect is the intended chain");
}

/// Two GPU frames of different backends. Neither the buffer variant nor a
/// single "is on a GPU" flag separates a D3D11 texture from a CUDA
/// allocation — only naming the backend does, which is why the domain is
/// an enum rather than a boolean.
#[cfg(all(target_os = "windows", feature = "d3d11", feature = "cuda"))]
#[test]
fn a_d3d11_texture_cannot_feed_a_cuda_filter() {
    use crate::elements::{CudaScaler, CudaScalerInterp, D3d11Upload};

    let Some((device, _d3d_context)) = crate::test_support::try_d3d11_device() else {
        eprintln!("skipped: no D3D11 hardware device available");
        return;
    };
    let Some((cuda, _cuda_guard)) = crate::test_support::try_cuda_device() else {
        return;
    };

    let Err(error) = contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .pipe(D3d11Upload::new("upload", &device, 64, 64))
        .pipe(CudaScaler::new(
            "scaler",
            &cuda,
            64,
            64,
            CudaScalerInterp::Bilinear,
        ))
        .to(DeclaringSink::boxed(
            "sink",
            InputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::Cuda,
            )),
        ))
    else {
        panic!("a D3D11 texture is not reachable from a CUDA kernel");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(&*producer, "upload");
    assert_eq!(&*consumer, "scaler");
}

/// `D3d12Renderer` takes device resources only, matching `D3d11Renderer`
/// and `CudaRenderer`. A CPU-decoded stream reaches it through
/// `D3d12Upload` — the one place a system frame crosses to the GPU —
/// rather than through a second upload path inside the sink.
#[cfg(all(target_os = "windows", feature = "d3d12"))]
#[test]
fn a_d3d12_renderer_takes_device_resources_only() {
    use std::any::Any;

    use windows::Win32::Graphics::Direct3D12::{ID3D12Device, ID3D12Fence, ID3D12Resource};

    use crate::elements::{D3d12FrameRenderer, D3d12Renderer, D3d12Upload, SubmitError};

    /// Never submitted to: the link check runs when the branch is built,
    /// so no buffer ever reaches these.
    struct StubRenderer(ID3D12Device);

    impl D3d12FrameRenderer for StubRenderer {
        fn device(&self) -> ID3D12Device {
            self.0.clone()
        }

        unsafe fn submit_nv12_texture(
            &self,
            _texture: ID3D12Resource,
            _fence: ID3D12Fence,
            _fence_value: u64,
            _width: u32,
            _height: u32,
            _keep_alive: Box<dyn Any + Send>,
        ) -> std::result::Result<(), SubmitError> {
            unreachable!("the link check never pushes a buffer")
        }

        fn resize(&self, _width: u32, _height: u32) -> std::result::Result<(), SubmitError> {
            unreachable!("the link check never resizes")
        }
    }

    let Some(device) = crate::test_support::try_d3d12_device() else {
        eprintln!("skipped: no D3D12 hardware device available");
        return;
    };
    // The second half of this test needs a working D3D12VA hw frames
    // context, which a device alone does not guarantee. Probe for it up
    // front so a machine without one skips rather than failing halfway.
    let upload = match D3d12Upload::new("upload", &device, 64, 64) {
        Ok(upload) => upload,
        Err(error) => {
            eprintln!("skipped: this D3D12 device cannot open a frames context ({error})");
            return;
        }
    };

    let Err(error) = contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .to(Box::new(D3d12Renderer::new(
            "renderer",
            Box::new(StubRenderer(device.clone())),
        )))
    else {
        panic!("a software decoder's frames have no path to the swap chain");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(&*producer, "decoder");
    assert_eq!(&*consumer, "renderer");

    // The upload that was missing is what makes the same chain valid.
    contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .pipe(upload)
        .to(Box::new(D3d12Renderer::new(
            "renderer",
            Box::new(StubRenderer(device)),
        )))
        .expect("a D3D12 resource is exactly what this renderer accepts");
}

/// A passthrough element that declared nothing used to end the check at
/// itself, because `Unknown` output means nothing is known to be flowing
/// onward. `VideoSynchronizer` sits mid-branch in every A/V playback
/// pipeline, so leaving it undeclared blinded the rest of the chain.
#[test]
fn a_video_synchronizer_carries_the_contract_past_itself() {
    use crate::elements::{PacketCounter, VideoSynchronizer};

    let context = contract_context();
    let sync = VideoSynchronizer::new("sync", ffmpeg::Rational::new(1, 90_000))
        .expect("a valid time base opens the synchronizer");

    let (counter, _count) = PacketCounter::new("counter");
    let Err(error) = context
        .branch()
        .pipe(video_decoder("decoder"))
        .pipe(sync)
        .to(Box::new(counter))
    else {
        panic!("scheduling frames does not turn them into packets");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(
        &*producer, "decoder",
        "a passthrough stage forwards the contract and the producer's name with it"
    );
    assert_eq!(&*consumer, "counter");
}

/// The head of a chain. A source pad that declares nothing leaves the
/// attach boundary unchecked for that whole pipeline, so the synthetic
/// and capture sources have to declare theirs too.
#[test]
fn a_video_source_cannot_be_attached_to_an_audio_branch() {
    use crate::elements::{TestVideoOptions, TestVideoSource};

    let context = contract_context();
    let branch = context
        .branch()
        .queue("q", 4)
        .to(DeclaringSink::boxed(
            "speakers",
            InputContract::Fixed(PortContract::frame(
                MediaKind::AudioFrame,
                MemoryDomain::System,
            )),
        ))
        .expect("the branch itself is consistent");

    let mut source = TestVideoSource::new("video", TestVideoOptions::default());
    let Err(error) = context.attach(&mut source, 0, branch) else {
        panic!("a video source has no samples for an audio renderer");
    };

    assert!(
        matches!(
            error,
            crate::Error::GraphError(GraphError::IncompatibleLink { .. })
        ),
        "got {error}"
    );
    assert!(
        !source.src_pads()[0].is_linked(),
        "a refused attach must not link the pad"
    );
}

/// The mistake the medium split exists for. A container's audio and video
/// pads both emit `MediaBuffer::Packet`, so before the kind carried the
/// medium this wired up cleanly and failed somewhere inside libavcodec on
/// the first packet instead.
#[test]
fn a_containers_audio_stream_cannot_feed_a_video_decoder() {
    let Some(path) = try_test_video() else { return };
    let (mut demuxer, streams) =
        FileDemuxer::open("demuxer", &path).expect("the fixture must open");
    eprintln!(
        "fixture streams: {:?}",
        streams
            .iter()
            .map(|s| (s.index, s.kind))
            .collect::<Vec<_>>()
    );
    let Some(audio) = streams
        .iter()
        .find(|s| s.kind == ffmpeg::media::Type::Audio)
    else {
        eprintln!("skipped: the fixture has no audio stream to mis-wire");
        return;
    };
    let audio_index = audio.index;
    let video = streams
        .iter()
        .find(|s| s.kind == ffmpeg::media::Type::Video)
        .expect("the fixture must have a video stream");
    let video_params = demuxer
        .stream_parameters(video.index)
        .expect("the video stream must expose parameters");

    let context = contract_context();
    let branch = context
        .branch()
        .pipe(SwDecoder::new("video-decoder", video_params).expect("the decoder must open"))
        .to(DeclaringSink::boxed("renderer", video_frames()))
        .expect("the branch itself is consistent");

    let Err(error) = context.attach(&mut demuxer, audio_index, branch) else {
        panic!("an audio stream has nothing a video decoder can decode");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        produced, accepted, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(produced.to_string(), "AudioPacket");
    assert_eq!(accepted.to_string(), "VideoPacket");

    // The video stream of the same container is what that decoder is for.
    let branch = context
        .branch()
        .pipe(
            SwDecoder::new(
                "video-decoder",
                demuxer
                    .stream_parameters(video.index)
                    .expect("the video stream must expose parameters"),
            )
            .expect("the decoder must open"),
        )
        .to(DeclaringSink::boxed("renderer", video_frames()))
        .expect("the branch itself is consistent");
    context
        .attach(&mut demuxer, video.index, branch)
        .expect("the video pad and a video decoder are exactly the intended link");
}

/// Guards the exact text README quotes, so the two cannot drift.
#[test]
fn an_incompatible_link_reads_the_way_the_readme_shows_it() {
    let (counter, _count) = crate::elements::PacketCounter::new("rec");
    let Err(error) = contract_context()
        .branch()
        .pipe(video_decoder("decoder"))
        .to(Box::new(counter))
    else {
        panic!("decoded frames are not packets");
    };
    assert_eq!(
        error.to_string(),
        "decoder produces VideoFrame (System), which rec cannot accept \
         (it takes VideoPacket|AudioPacket)"
    );
}

/// A `D3d11FrameRenderer` that is never submitted to: the link check runs
/// when a branch is built or attached, so no buffer ever reaches it.
#[cfg(all(target_os = "windows", feature = "d3d11"))]
struct StubD3d11Renderer(windows::Win32::Graphics::Direct3D11::ID3D11Device);

#[cfg(all(target_os = "windows", feature = "d3d11"))]
impl crate::elements::D3d11FrameRenderer for StubD3d11Renderer {
    fn device(&self) -> windows::Win32::Graphics::Direct3D11::ID3D11Device {
        self.0.clone()
    }

    unsafe fn submit_bgra_texture(
        &self,
        _texture: windows::Win32::Graphics::Direct3D11::ID3D11Texture2D,
        _array_index: u32,
        _width: u32,
        _height: u32,
    ) -> std::result::Result<(), crate::elements::SubmitError> {
        unreachable!("the link check never pushes a buffer")
    }

    unsafe fn submit_nv12_texture(
        &self,
        _texture: windows::Win32::Graphics::Direct3D11::ID3D11Texture2D,
        _array_index: u32,
        _width: u32,
        _height: u32,
    ) -> std::result::Result<(), crate::elements::SubmitError> {
        unreachable!("the link check never pushes a buffer")
    }

    fn resize(
        &self,
        _width: u32,
        _height: u32,
    ) -> std::result::Result<(), crate::elements::SubmitError> {
        unreachable!("the link check never resizes")
    }
}

/// A branch whose leading stages are all passthrough has no requirement of
/// its own — the one that matters belongs to an element further down, and
/// summarizing the branch as "what its first stage accepts" threw that
/// away. `VideoSynchronizer` takes a frame from any backend, so before the
/// branch was re-walked at attach time a system-memory source linked to a
/// D3D11 renderer cleanly and failed per frame at runtime.
#[cfg(all(target_os = "windows", feature = "d3d11"))]
#[test]
fn a_passthrough_at_the_head_of_a_branch_still_carries_the_downstream_requirement() {
    use crate::elements::{TestVideoOptions, TestVideoSource, VideoSynchronizer};

    let Some((device, d3d_context)) = crate::test_support::try_d3d11_device() else {
        eprintln!("skipped: no D3D11 hardware device available");
        return;
    };
    let context = contract_context();
    let renderer = crate::elements::D3d11Renderer::new(
        "renderer",
        Box::new(StubD3d11Renderer(device.clone())),
    );
    let _ = &d3d_context;

    let branch = context
        .branch()
        .pipe(
            VideoSynchronizer::new("sync", ffmpeg::Rational::new(1, 90_000))
                .expect("a valid time base opens the synchronizer"),
        )
        .to(Box::new(renderer))
        .expect("nothing is flowing yet, so the branch alone is consistent");
    let before = context.graph.snapshot();

    let mut source = TestVideoSource::new("video", TestVideoOptions::default());
    let Err(error) = context.attach(&mut source, 0, branch) else {
        panic!("system-memory frames never reach a D3D11 swap chain");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(
        &*producer, "video_src",
        "the synchronizer forwards the pad's own contract rather than replacing it"
    );
    assert_eq!(
        &*consumer, "renderer",
        "the requirement belongs to the element past the passthrough stage"
    );

    assert!(!source.src_pads()[0].is_linked());
    let after = context.graph.snapshot();
    assert_eq!(before.revision, after.revision);
    assert_eq!(before.nodes.len(), after.nodes.len());
}

/// The same shape with the upload that was missing, to prove the walk is
/// not simply refusing every branch that starts with a passthrough stage.
#[cfg(all(target_os = "windows", feature = "d3d11"))]
#[test]
fn a_passthrough_at_the_head_of_a_branch_accepts_a_matching_source() {
    use crate::elements::{D3d11Upload, TestVideoOptions, TestVideoSource, VideoSynchronizer};

    let Some((device, _d3d_context)) = crate::test_support::try_d3d11_device() else {
        eprintln!("skipped: no D3D11 hardware device available");
        return;
    };
    let context = contract_context();
    let renderer = crate::elements::D3d11Renderer::new(
        "renderer",
        Box::new(StubD3d11Renderer(device.clone())),
    );

    let branch = context
        .branch()
        .pipe(
            VideoSynchronizer::new("sync", ffmpeg::Rational::new(1, 90_000))
                .expect("a valid time base opens the synchronizer"),
        )
        .pipe(D3d11Upload::new("upload", &device, 64, 64))
        .to(Box::new(renderer))
        .expect("the branch itself is consistent");

    let mut source = TestVideoSource::new("video", TestVideoOptions::default());
    context
        .attach(&mut source, 0, branch)
        .expect("the upload is exactly what makes this source reach the renderer");
}

// ---------------------------------------------------------------------
// Tee branches, initial and dynamic.
//
// A Tee's pads are Passthrough, so nothing about the pad alone says what
// a branch attached to it will receive. Both of these used to link
// unchecked: the initial branches because their plans were merged into
// the Tee's without their contracts, and the dynamic ones because
// TeeHandle::attach committed straight to the graph.
// ---------------------------------------------------------------------

fn audio_sink(name: &'static str) -> Box<DeclaringSink> {
    DeclaringSink::boxed(
        name,
        InputContract::Fixed(PortContract::frame(
            MediaKind::AudioFrame,
            MemoryDomain::System,
        )),
    )
}

/// An initial branch is merged into the `Tee`'s own plan, so the attach
/// that commits the `Tee` has to check it against what the `Tee` receives.
#[test]
fn an_initial_tee_branch_is_checked_against_what_the_tee_receives() {
    use crate::elements::{TestVideoOptions, TestVideoSource};

    let context = contract_context();
    let audio_branch = context
        .branch()
        .to(audio_sink("speakers"))
        .expect("the branch itself is consistent");
    let tee = TeeBuilder::new("tee", context.clone())
        .branch(audio_branch)
        .build()
        .expect("building the fan-out does not check it against a source");
    let before = context.graph.snapshot();

    let mut source = TestVideoSource::new("video", TestVideoOptions::default());
    let Err(error) = context.attach(&mut source, 0, tee) else {
        panic!("a video source has no samples for an audio sink behind a Tee");
    };

    let crate::Error::GraphError(GraphError::IncompatibleLink {
        producer, consumer, ..
    }) = error
    else {
        panic!("expected an IncompatibleLink, got {error}");
    };
    assert_eq!(
        &*producer, "video_src",
        "the Tee forwards what it was given rather than producing its own"
    );
    assert_eq!(&*consumer, "speakers");

    assert!(!source.src_pads()[0].is_linked());
    let after = context.graph.snapshot();
    assert_eq!(before.revision, after.revision);
    assert_eq!(before.nodes.len(), after.nodes.len());
}

/// Valid siblings all attach — the fan-out edge hands each branch the same
/// flow, so a Tee with several good branches is not refused.
#[test]
fn every_valid_initial_tee_branch_attaches() {
    use crate::elements::{TestVideoOptions, TestVideoSource};

    let context = contract_context();
    let one = context
        .branch()
        .to(DeclaringSink::boxed("first", video_frames()))
        .expect("consistent");
    let two = context
        .branch()
        .queue("q", 4)
        .to(DeclaringSink::boxed("second", video_frames()))
        .expect("consistent");
    let tee = TeeBuilder::new("tee", context.clone())
        .branch(one)
        .branch(two)
        .build()
        .expect("consistent");

    let mut source = TestVideoSource::new("video", TestVideoOptions::default());
    context
        .attach(&mut source, 0, tee)
        .expect("both branches take the frames this source produces");
}

/// A branch added while the pipeline runs is checked against the flow its
/// siblings already carry, which the graph recorded when the `Tee` was
/// committed. A failure leaves the fan-out exactly as it was.
#[test]
fn a_dynamic_tee_branch_is_checked_and_a_refusal_changes_nothing() {
    use crate::elements::{TestVideoOptions, TestVideoSource};

    let context = contract_context();
    let initial = context
        .branch()
        .to(DeclaringSink::boxed("renderer", video_frames()))
        .expect("consistent");
    let (tee, handle) = TeeBuilder::new("tee", context.clone())
        .branch(initial)
        .build_dynamic()
        .expect("consistent");
    let mut source = TestVideoSource::new("video", TestVideoOptions::default());
    context
        .attach(&mut source, 0, tee)
        .expect("the initial fan-out matches the source");

    let attached = context.graph.snapshot();
    let audio_branch = handle
        .branch()
        .expect("the Tee is alive")
        .to(audio_sink("speakers"))
        .expect("the branch itself is consistent");

    let Err(error) = handle.attach(audio_branch) else {
        panic!("this Tee carries video frames, not samples");
    };
    assert!(
        matches!(
            error,
            crate::Error::GraphError(GraphError::IncompatibleLink { .. })
        ),
        "got {error}"
    );

    let after = context.graph.snapshot();
    assert_eq!(
        attached.revision, after.revision,
        "a refused dynamic attach must not bump the graph revision"
    );
    assert_eq!(attached.nodes.len(), after.nodes.len());
    assert_eq!(attached.edges.len(), after.edges.len());

    // The Tee still works, and a matching branch still attaches.
    let good = handle
        .branch()
        .expect("the Tee is alive")
        .to(DeclaringSink::boxed("second", video_frames()))
        .expect("consistent");
    handle
        .attach(good)
        .expect("a video branch is what this Tee can feed");
}

/// Resolved contracts are live-graph state just like nodes and edges. A
/// detached dynamic branch must release those entries too; element IDs are
/// never reused, so leaving one behind would grow the graph for every churn
/// cycle even though snapshots and `sink_count` said the branch was gone.
#[test]
fn detaching_a_dynamic_tee_branch_releases_its_resolved_contracts() {
    use crate::elements::{TestVideoOptions, TestVideoSource};

    let context = contract_context();
    let fixed = context
        .branch()
        .to(DeclaringSink::boxed("fixed", video_frames()))
        .expect("consistent");
    let (tee, handle) = TeeBuilder::new("tee", context.clone())
        .branch(fixed)
        .build_dynamic()
        .expect("consistent");
    let mut source = TestVideoSource::new("video", TestVideoOptions::default());
    context
        .attach(&mut source, 0, tee)
        .expect("the fixed branch matches the source");
    let baseline = context.graph.resolved_output_count();

    let dynamic = handle
        .branch()
        .expect("the Tee is alive")
        .queue("dynamic-q", 4)
        .to(DeclaringSink::boxed("dynamic", video_frames()))
        .expect("consistent");
    let branch_id = handle.attach(dynamic).expect("attach the dynamic branch");
    assert!(
        context.graph.resolved_output_count() > baseline,
        "the attached branch must contribute resolved contract entries"
    );

    handle.detach(branch_id).expect("detach the dynamic branch");
    assert_eq!(
        context.graph.resolved_output_count(),
        baseline,
        "detaching must release every resolved contract owned by the branch"
    );
}
