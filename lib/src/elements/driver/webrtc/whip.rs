//! Publishing to a WHIP server: [`WebRtcMuxer`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ffmpeg_next as ffmpeg;
use str0m::Rtc;
use str0m::change::SdpAnswer;
use str0m::format::Codec;
use str0m::media::{Direction, MediaKind};

use crate::element::{ElementType, RawSink, element_pp_log};
use crate::elements::sink::MuxerId;
use crate::elements::{MuxerSinks, MuxerTrack, StreamInfo};
use crate::error::Result;
use crate::pp_log::{PpLog, pp_info};

use super::client::{self, Client, Parts, Url, WebRtcOptions};
use super::command::{OpenTrack, TrackId, WebRtcError};
use super::outbox::Outbox;
use super::stream_info::str0m_codec;
use super::track::WebRtcHandle;
use super::track::WebRtcTrackSink;

/// How many packets the connection holds for its tracks before it is taken
/// to have fallen behind — see [`WebRtcTrackSink`].
const OUTBOX: usize = 512;

/// Publishes encoded tracks to a WHIP server — `WebRtcServer`, MediaMTX,
/// a streaming service that takes WHIP. Needs the `webrtc` feature.
///
/// Made, given its tracks, and opened, as an
/// [`RtspMuxer`](crate::elements::RtspMuxer) is: [`Self::open`] posts the
/// offer, takes the answer, and waits for the connection, so a server
/// that refuses — a wrong token, a path it does not serve, one already
/// published — or cannot be reached is an error there. It hands out a sink
/// for each track, as every muxer does, to end each track's chain with.
/// What [`WebRtcSource`](crate::elements::WebRtcSource) is to receiving,
/// this is to sending.
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::elements::{StreamInfo, WebRtcMuxer, WebRtcOptions};
/// # let video: StreamInfo = unimplemented!();
///
/// let mut muxer = WebRtcMuxer::create("camera", "https://example.com/live/whip", WebRtcOptions {
///     bearer_token: Some("stream-key".into()),
///     ..WebRtcOptions::default()
/// })?;
/// let track = muxer.add_stream("video", video)?; // an H.264 encoder's stream
/// let mut sinks = muxer.open()?;
/// let sink = sinks.take(track)?; // ends the encoder's chain
/// # Ok(())
/// # }
/// ```
///
/// # How it works
///
/// One WebRTC connection, with a thread of its own driving it. Its tracks
/// carry what WebRTC does — H.264 without B-frames, H.265, VP8, VP9, AV1,
/// Opus, G.711 — each in the form the track's parameters say, an MP4's
/// `avcC` made Annex B on the way. The connection lasts as long as its
/// sinks: the last of them dropped ends the session on the server with a
/// `DELETE`, closes the connection and joins its thread. A connection that
/// falls behind goes on from each track's next keyframe.
pub struct WebRtcMuxer {
    name: Arc<str>,
    via: Via,
    /// Each track: its name, what it carries, and its parameters.
    tracks: Vec<(Arc<str>, MediaKind, ffmpeg::codec::Parameters)>,
    id: MuxerId,
    pp_log: PpLog,
}

/// What a [`WebRtcMuxer`] sends over.
enum Via {
    /// A connection of its own, made with a WHIP server as it opens.
    Whip { url: Url, options: WebRtcOptions },
    /// A peer's, which it adds its tracks to as it opens.
    Peer(WebRtcHandle),
}

// SAFETY: the codec parameters are this value's own copies, not shared with
// anything else, and moved with it rather than used from two threads.
unsafe impl Send for WebRtcMuxer {}

impl WebRtcMuxer {
    /// A publisher to `url` with no tracks yet. Nothing is sent before
    /// [`Self::open`].
    ///
    /// # Errors
    ///
    /// [`WebRtcError::InvalidUrl`] for a URL that is not `http` or `https`
    /// with a host.
    pub fn create(name: impl Into<String>, url: &str, options: WebRtcOptions) -> Result<Self> {
        let name: Arc<str> = name.into().into();
        let url = Url::parse(url)?;
        let pp_log = element_pp_log(ElementType::WebRtcMuxer, &name, None);
        Ok(Self {
            name,
            via: Via::Whip { url, options },
            tracks: Vec::new(),
            id: MuxerId::next(),
            pp_log,
        })
    }

    /// A muxer of `handle`'s peer's connection — see
    /// [`WebRtcHandle::muxer`].
    pub(super) fn over_peer(name: impl Into<String>, handle: WebRtcHandle) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::WebRtcMuxer, &name, None);
        Self {
            name,
            via: Via::Peer(handle),
            tracks: Vec::new(),
            id: MuxerId::next(),
            pp_log,
        }
    }

    /// Adds a track called `name` of what `format` describes — an
    /// encoder's stream, or a demuxer's — and answers what its sink is
    /// taken out of [`Self::open`]'s answer with.
    ///
    /// # Errors
    ///
    /// [`WebRtcError::SourceCodecUnsupported`] for a codec WebRTC does not
    /// carry — AAC among them.
    pub fn add_stream(
        &mut self,
        name: impl Into<String>,
        format: impl Into<StreamInfo>,
    ) -> Result<MuxerTrack> {
        let parameters = format.into().into_parameters();
        let id = parameters.id();
        let kind = match (str0m_codec(id), parameters.medium()) {
            (Some(_), ffmpeg::media::Type::Video) => MediaKind::Video,
            (Some(_), ffmpeg::media::Type::Audio) => MediaKind::Audio,
            _ => return Err(WebRtcError::SourceCodecUnsupported(id).into()),
        };
        let track = self.id.track(self.tracks.len());
        self.tracks.push((name.into().into(), kind, parameters));
        Ok(track)
    }

    /// Posts the offer, takes the answer, and waits for the connection:
    /// the tracks' sinks, once it is up, each taken out by the
    /// [`MuxerTrack`] its [`Self::add_stream`] returned.
    ///
    /// # Errors
    ///
    /// [`WebRtcError::NoTracks`] with none added;
    /// [`WebRtcError::Request`] where the server cannot be reached or the
    /// connection fails; [`WebRtcError::Refused`] with the status the
    /// server refused with; [`WebRtcError::NotConnected`] where the
    /// connection does not come up within
    /// [`WebRtcOptions::connect_within`]; and
    /// [`WebRtcError::OutboundCodecNotNegotiated`] for a track whose codec
    /// the server's answer did not take. Nothing is left open on an error.
    pub fn open(self) -> Result<MuxerSinks> {
        if self.tracks.is_empty() {
            return Err(WebRtcError::NoTracks.into());
        }
        let (url, options) = match &self.via {
            Via::Whip { url, options } => (url, options),
            Via::Peer(handle) => return self.open_over_peer(handle),
        };
        let (socket, local) = client::socket_toward(url)?;
        let mut rtc = Rtc::builder().build(Instant::now());
        rtc.add_local_candidate(client::candidate(local)?);
        let mut changes = rtc.sdp_api();
        let mids: Vec<_> = self
            .tracks
            .iter()
            .map(|(_, kind, _)| changes.add_media(*kind, Direction::SendOnly, None, None, None))
            .collect();
        let (offer, pending) = changes.apply().ok_or_else(|| WebRtcError::Request {
            url: url.full.clone(),
            reason: "there was nothing to offer".into(),
        })?;
        let (answer, resource) = client::exchange(
            url,
            options.bearer_token.as_deref(),
            &offer.to_sdp_string(),
            options.connect_within,
        )?;
        let accepted = SdpAnswer::from_sdp_string(&answer)
            .map_err(|error| WebRtcError::Request {
                url: url.full.clone(),
                reason: format!("its answer is not SDP: {error}"),
            })
            .and_then(|answer| {
                rtc.sdp_api()
                    .accept_answer(pending, answer)
                    .map_err(WebRtcError::from)
            });
        let codecs: Vec<Vec<Codec>> = mids
            .iter()
            .map(|mid| {
                let mut codecs = Vec::new();
                if let Some(writer) = rtc.writer(*mid) {
                    for codec in writer.payload_params().map(|params| params.spec().codec) {
                        if !codecs.contains(&codec) {
                            codecs.push(codec);
                        }
                    }
                }
                codecs
            })
            .collect();
        let outbox = Outbox::new(OUTBOX);
        // Made even where the answer was refused, so that ending it ends the
        // session the server made.
        let (client, said) = Client::spawn(Parts {
            name: self.name.to_string(),
            rtc,
            socket,
            local,
            outgoing: mids
                .iter()
                .enumerate()
                .map(|(index, mid)| (TrackId(index as u64), *mid))
                .collect::<HashMap<_, _>>(),
            outbox: Some(outbox.clone()),
            incoming: Vec::new(),
            resource,
            pp_log: self.pp_log.clone(),
        })?;
        accepted?;
        client::connected(&said, url, options.connect_within)?;
        pp_info!(
            pp_log: &self.pp_log,
            "publishing {} tracks to {}",
            self.tracks.len(),
            url.full
        );
        let keeps: Arc<dyn std::any::Any + Send + Sync> = client;
        let mut sinks: Vec<Box<dyn RawSink>> = Vec::with_capacity(self.tracks.len());
        for (index, ((name, kind, parameters), codecs)) in
            self.tracks.iter().zip(codecs).enumerate()
        {
            let mut sink = WebRtcTrackSink::new(
                TrackId(index as u64),
                *kind,
                None,
                Arc::new(Mutex::new(codecs)),
                outbox.clone(),
            )
            .keeping(Arc::clone(&keeps))
            .named(ElementType::WebRtcMuxer, Arc::clone(name));
            sink.set_source_parameters(parameters)?;
            sinks.push(Box::new(sink));
        }
        Ok(self.id.sinks(sinks))
    }

    /// Has the peer add the tracks, sending only, in one renegotiation, and
    /// answers their sinks once the offer is made.
    fn open_over_peer(&self, handle: &WebRtcHandle) -> Result<MuxerSinks> {
        let tracks = self
            .tracks
            .iter()
            .map(|(name, kind, parameters)| {
                let codec = str0m_codec(parameters.id())
                    .ok_or(WebRtcError::SourceCodecUnsupported(parameters.id()))?;
                Ok(OpenTrack {
                    id: handle.track_id(),
                    kind: *kind,
                    codec,
                    name: Arc::clone(name),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let opened = handle.open_tracks(tracks)?;
        pp_info!(
            pp_log: &self.pp_log,
            "{} tracks offered over the peer's connection",
            self.tracks.len()
        );
        let mut sinks: Vec<Box<dyn RawSink>> = Vec::with_capacity(opened.len());
        for (sink, (name, _, parameters)) in opened.into_iter().zip(&self.tracks) {
            let mut sink = sink.named(ElementType::WebRtcMuxer, Arc::clone(name));
            sink.set_source_parameters(parameters)?;
            sinks.push(Box::new(sink));
        }
        Ok(self.id.sinks(sinks))
    }
}
