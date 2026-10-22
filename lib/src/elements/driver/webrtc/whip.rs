//! Publishing to a WHIP server: [`WhipPublisher`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ffmpeg_next as ffmpeg;
use str0m::Rtc;
use str0m::change::SdpAnswer;
use str0m::format::Codec;
use str0m::media::{Direction, MediaKind};

use crate::element::{ElementType, element_pp_log};
use crate::error::Result;
use crate::pp_log::{PpLog, pp_info};

use super::client::{self, Client, Parts, Url};
use super::command::{TrackId, WebRtcError};
use super::outbox::Outbox;
use super::stream_info::str0m_codec;
use super::track::WebRtcTrackSink;

/// How many packets the connection holds for its tracks before it is taken
/// to have fallen behind — see [`WebRtcTrackSink`].
const OUTBOX: usize = 512;

/// How a [`WhipPublisher`] reaches its server.
#[derive(Debug, Clone)]
pub struct WhipOptions {
    /// Sent as `Authorization: Bearer <token>`, where the server asks for
    /// one — a stream key, as WHIP servers call it.
    pub bearer_token: Option<String>,
    /// How long [`WhipPublisher::open`] waits for the server's answer, and
    /// then for the connection to come up. Ten seconds by default.
    pub connect_within: Duration,
}

impl Default for WhipOptions {
    fn default() -> Self {
        Self {
            bearer_token: None,
            connect_within: Duration::from_secs(10),
        }
    }
}

/// Publishes encoded tracks to a WHIP server — `WebRtcServer`, MediaMTX,
/// a streaming service that takes WHIP. Needs the `webrtc` feature.
///
/// Made, given its tracks, and opened, as an
/// [`RtspMuxer`](crate::elements::RtspMuxer) is: [`Self::open`] posts the
/// offer, takes the answer, and waits for the connection, so a server
/// that refuses — a wrong token, a path it does not serve, one already
/// published — or cannot be reached is an error there. It hands out a
/// [`WebRtcTrackSink`] for each track, the sink a
/// [`WebRtcPeer`](crate::elements::WebRtcPeer) hands out, to end each
/// track's chain with.
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::elements::{WhipOptions, WhipPublisher};
/// # let video: media_pp::ffmpeg::codec::Parameters = unimplemented!();
///
/// let mut publisher = WhipPublisher::create("camera", "https://example.com/live/whip", WhipOptions {
///     bearer_token: Some("stream-key".into()),
///     ..WhipOptions::default()
/// })?;
/// let track = publisher.add_track(&video)?; // an H.264 encoder's parameters
/// let mut sinks = publisher.open()?;
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
pub struct WhipPublisher {
    name: Arc<str>,
    url: Url,
    options: WhipOptions,
    tracks: Vec<(MediaKind, ffmpeg::codec::Parameters)>,
    pp_log: PpLog,
}

// SAFETY: the codec parameters are this value's own copies, not shared with
// anything else, and moved with it rather than used from two threads.
unsafe impl Send for WhipPublisher {}

/// A [`WhipPublisher`]'s tracks, each a sink to end its chain with.
pub struct WhipSinks {
    sinks: Vec<Option<WebRtcTrackSink>>,
}

impl WhipPublisher {
    /// A publisher to `url` with no tracks yet. Nothing is sent before
    /// [`Self::open`].
    ///
    /// # Errors
    ///
    /// [`WebRtcError::InvalidUrl`] for a URL that is not `http` or `https`
    /// with a host.
    pub fn create(name: impl Into<String>, url: &str, options: WhipOptions) -> Result<Self> {
        let name: Arc<str> = name.into().into();
        let url = Url::parse(url)?;
        let pp_log = element_pp_log(ElementType::WhipPublisher, &name, None);
        Ok(Self {
            name,
            url,
            options,
            tracks: Vec::new(),
            pp_log,
        })
    }

    /// Adds a track of what `parameters` describe — an encoder's, or a
    /// demuxer's stream's — and answers its number, which
    /// [`WhipSinks::take`] takes.
    ///
    /// # Errors
    ///
    /// [`WebRtcError::SourceCodecUnsupported`] for a codec WebRTC does not
    /// carry — AAC among them.
    pub fn add_track(&mut self, parameters: &ffmpeg::codec::Parameters) -> Result<usize> {
        let id = parameters.id();
        let kind = match (str0m_codec(id), parameters.medium()) {
            (Some(_), ffmpeg::media::Type::Video) => MediaKind::Video,
            (Some(_), ffmpeg::media::Type::Audio) => MediaKind::Audio,
            _ => return Err(WebRtcError::SourceCodecUnsupported(id).into()),
        };
        self.tracks.push((kind, parameters.clone()));
        Ok(self.tracks.len() - 1)
    }

    /// Posts the offer, takes the answer, and waits for the connection:
    /// the tracks' sinks, once it is up.
    ///
    /// # Errors
    ///
    /// [`WebRtcError::NoTracks`] with none added;
    /// [`WebRtcError::Request`] where the server cannot be reached or the
    /// connection fails; [`WebRtcError::Refused`] with the status the
    /// server refused with; [`WebRtcError::NotConnected`] where the
    /// connection does not come up within
    /// [`WhipOptions::connect_within`]; and
    /// [`WebRtcError::OutboundCodecNotNegotiated`] for a track whose codec
    /// the server's answer did not take. Nothing is left open on an error.
    pub fn open(self) -> Result<WhipSinks> {
        if self.tracks.is_empty() {
            return Err(WebRtcError::NoTracks.into());
        }
        let (socket, local) = client::socket_toward(&self.url)?;
        let mut rtc = Rtc::builder().build(Instant::now());
        rtc.add_local_candidate(client::candidate(local)?);
        let mut changes = rtc.sdp_api();
        let mids: Vec<_> = self
            .tracks
            .iter()
            .map(|(kind, _)| changes.add_media(*kind, Direction::SendOnly, None, None, None))
            .collect();
        let (offer, pending) = changes.apply().ok_or_else(|| WebRtcError::Request {
            url: self.url.full.clone(),
            reason: "there was nothing to offer".into(),
        })?;
        let (answer, resource) = client::exchange(
            &self.url,
            self.options.bearer_token.as_deref(),
            &offer.to_sdp_string(),
            self.options.connect_within,
        )?;
        let accepted = SdpAnswer::from_sdp_string(&answer)
            .map_err(|error| WebRtcError::Request {
                url: self.url.full.clone(),
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
        client::connected(&said, &self.url, self.options.connect_within)?;
        pp_info!(
            pp_log: &self.pp_log,
            "publishing {} tracks to {}",
            self.tracks.len(),
            self.url.full
        );
        let keeps: Arc<dyn std::any::Any + Send + Sync> = client;
        let mut sinks = Vec::with_capacity(self.tracks.len());
        for (index, ((kind, parameters), codecs)) in self.tracks.iter().zip(codecs).enumerate() {
            let mut sink = WebRtcTrackSink::new(
                TrackId(index as u64),
                *kind,
                None,
                Arc::new(Mutex::new(codecs)),
                outbox.clone(),
            )
            .keeping(Arc::clone(&keeps));
            sink.set_source_parameters(parameters)?;
            sinks.push(Some(sink));
        }
        Ok(WhipSinks { sinks })
    }
}

impl WhipSinks {
    /// The sink of track `track`, as [`WhipPublisher::add_track`] numbered
    /// it — each once.
    ///
    /// # Errors
    ///
    /// [`WebRtcError::NoSuchTrack`] for a track it does not have, or one
    /// already taken.
    pub fn take(&mut self, track: usize) -> Result<WebRtcTrackSink> {
        self.sinks
            .get_mut(track)
            .and_then(Option::take)
            .ok_or_else(|| WebRtcError::NoSuchTrack(track).into())
    }
}
