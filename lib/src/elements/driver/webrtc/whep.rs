//! Playing from a WHEP server: [`WebRtcSource`].

use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long a source of a peer's tracks waits for them to say what they
/// carry, as [`WebRtcOptions::describe_within`] does by default for one
/// made over WHEP.
const DESCRIBE_WITHIN: Duration = Duration::from_secs(5);

use str0m::Rtc;
use str0m::change::SdpAnswer;
use str0m::media::{Direction, MediaKind};

use crate::contract::{OutputContract, PortContract};
use crate::element::{
    Attached, ElementType, Identity, Produced, Source, SourceStage, Wait, element_pp_log,
};
use crate::elements::StreamInfo;
use crate::error::Result;
use crate::pad::SrcPad;
use crate::pp_log::{PpLog, pp_info, pp_warn};
use crate::produce::{Received, source_stage};

use super::client::{self, Client, Incoming, Parts, Url, WebRtcOptions};
use super::command::WebRtcError;
use super::inbox::{Inbox, Taken};
use super::peer::Arriving;
use super::stream_info::WebRtcStreamInfo;

/// How many packets a track holds for the source before the newest are
/// let go of.
const INBOX: usize = 1024;

/// Plays a stream from a WHEP server — `WebRtcServer`, MediaMTX, any server
/// that speaks WHEP — into a pipeline: an output for each of its tracks,
/// handing on their packets as they arrive. Needs the `webrtc` feature.
///
/// Opened as an [`RtspSource`](crate::elements::RtspSource) is: on a URL,
/// answering with the streams it carries, so the pipeline after it is
/// built from what is known — a decoder, a muxer, a
/// [`Broadcast`](crate::elements::Broadcast) to relay it through.
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::elements::{FrameCounter, SwDecoder, WebRtcOptions, WebRtcSource};
/// use media_pp::pipeline::Pipeline;
///
/// let (source, streams) = WebRtcSource::open("relay", "https://example.com/cam1/whep", WebRtcOptions::default())?;
/// let video = streams.iter().find(|s| s.kind() == media_pp::ffmpeg::media::Type::Video).unwrap();
/// let decoder = SwDecoder::new("decoder", video.parameters().clone())?;
/// let (counter, _pictures) = FrameCounter::new("pictures");
/// let (pipeline, ()) = Pipeline::new("play", source, |source, ctx| {
///     ctx.attach(source, video.index(), ctx.branch().pipe(decoder).to(counter)?)?;
///     Ok(())
/// })?;
/// pipeline.run()?;
/// # Ok(())
/// # }
/// ```
///
/// # How it works
///
/// It offers to receive a picture and a sound, and its outputs are the
/// tracks the server sent — in that order, a picture first — whose packets
/// said what they carry within [`WebRtcOptions::describe_within`]: the
/// packets come stamped in their RTP clock, which each [`StreamInfo`]'s
/// time base is. A thread of its own drives the connection and fills each
/// track's inbox, which the source reads: it can be driven — run in a
/// [`DrivenPipeline`](crate::pipeline::DrivenPipeline) — since a packet
/// wakes the poll. The server ending the session, or the connection lost,
/// ends its stream; dropped, it ends the session on the server with a
/// `DELETE`.
pub struct WebRtcSource(SourceStage<Watching>);

source_stage!(WebRtcSource);

/// What a [`WebRtcSource`] does: hands on what its connection receives, a
/// track an output.
struct Watching {
    inboxes: Vec<Inbox>,
    contracts: Vec<OutputContract>,
    /// Which output is looked at first next, so that one busy track does
    /// not keep another waiting.
    next: usize,
    /// The connection, where this keeps it alive: a WHEP one, which ends
    /// with its source; not a peer's, which runs on its own.
    _client: Option<Arc<Client>>,
    pp_log: PpLog,
}

impl WebRtcSource {
    /// Connects to `url` and waits for its tracks to say what they carry:
    /// the source, and its streams — each the output of its `index`.
    ///
    /// # Errors
    ///
    /// [`WebRtcError::InvalidUrl`], [`WebRtcError::Request`] where the
    /// server cannot be reached or the connection fails,
    /// [`WebRtcError::Refused`] with the status it refused with,
    /// [`WebRtcError::NotConnected`], and [`WebRtcError::NothingDescribed`]
    /// where no track said what it carries in time. Nothing is left open
    /// on an error.
    pub fn open(
        name: impl Into<String>,
        url: &str,
        options: WebRtcOptions,
    ) -> Result<(Self, Vec<StreamInfo>)> {
        let name = name.into();
        let url = Url::parse(url)?;
        let pp_log = element_pp_log(ElementType::WebRtcSource, &name, None);
        let (socket, local) = client::socket_toward(&url)?;
        let mut rtc = Rtc::builder().build(Instant::now());
        rtc.add_local_candidate(client::candidate(local)?);
        let mut changes = rtc.sdp_api();
        let mids = [
            changes.add_media(MediaKind::Video, Direction::RecvOnly, None, None, None),
            changes.add_media(MediaKind::Audio, Direction::RecvOnly, None, None, None),
        ];
        let (offer, pending) = changes.apply().ok_or_else(|| WebRtcError::Request {
            url: url.full.clone(),
            reason: "there was nothing to offer".into(),
        })?;
        let (answer, resource) = client::exchange(
            &url,
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
        let inboxes = Inbox::group(mids.len(), INBOX);
        // Made even where the answer was refused, so that ending it ends the
        // session the server made.
        let (client, said) = Client::spawn(Parts {
            name: name.clone(),
            rtc,
            socket,
            local,
            outgoing: Default::default(),
            outbox: None,
            incoming: mids
                .iter()
                .zip(&inboxes)
                .map(|(mid, inbox)| Incoming {
                    mid: *mid,
                    inbox: inbox.clone(),
                })
                .collect(),
            resource,
            pp_log: pp_log.clone(),
        })?;
        accepted?;
        client::connected(&said, &url, options.connect_within)?;

        // Each track says what it is in its own time; a server may have
        // nothing to send on one of them at all.
        let mut described = vec![None; mids.len()];
        let deadline = Instant::now() + options.describe_within;
        while described.iter().any(Option::is_none) {
            let left = deadline.saturating_duration_since(Instant::now());
            let Ok((mid, info)) = said.described.recv_timeout(left) else {
                break;
            };
            if let Some(slot) = mids.iter().position(|known| *known == mid) {
                described[slot] = Some(info);
            }
        }
        let (source, streams) = Self::described(
            name,
            inboxes.into_iter().zip(described).collect(),
            options.describe_within,
            Some(client),
            pp_log,
        )?;
        pp_info!(pp_log: &source.0.inner().pp_log, "playing {} tracks from {}", streams.len(), url.full);
        Ok((source, streams))
    }

    /// A source of tracks a peer's remote side added together — see
    /// [`WebRtcHandle::next_source`](super::WebRtcHandle::next_source).
    pub(super) fn arriving(
        name: String,
        arriving: Vec<Arriving>,
    ) -> Result<(Self, Vec<StreamInfo>)> {
        let pp_log = element_pp_log(ElementType::WebRtcSource, &name, None);
        let deadline = Instant::now() + DESCRIBE_WITHIN;
        let tracks = arriving
            .into_iter()
            .map(|track| {
                let left = deadline.saturating_duration_since(Instant::now());
                let info = track.described.recv_timeout(left).ok();
                (track.inbox, info)
            })
            .collect();
        let (source, streams) = Self::described(name, tracks, DESCRIBE_WITHIN, None, pp_log)?;
        pp_info!(
            pp_log: &source.0.inner().pp_log,
            "playing {} tracks the remote peer added",
            streams.len()
        );
        Ok((source, streams))
    }

    /// The source of `tracks` that said what they carry — an output each,
    /// in order — and their streams; the others' inboxes let go of.
    fn described(
        name: String,
        tracks: Vec<(Inbox, Option<WebRtcStreamInfo>)>,
        within: Duration,
        client: Option<Arc<Client>>,
        pp_log: PpLog,
    ) -> Result<(Self, Vec<StreamInfo>)> {
        let offered = tracks.len();
        let mut streams = Vec::new();
        let mut kept = Vec::new();
        let mut contracts = Vec::new();
        for (inbox, info) in tracks {
            let Some(info) = info else {
                // Nothing will read it: what comes is let go of.
                inbox.reader_gone();
                continue;
            };
            let parameters = info.codec_parameters()?;
            let kind = parameters.medium();
            contracts.push(
                crate::contract::MediaKind::packet_for(kind)
                    .map_or(OutputContract::Unknown, |kind| {
                        OutputContract::Fixed(PortContract::packet(kind))
                    }),
            );
            streams.push(StreamInfo::new(parameters, info.time_base()?).with_index(streams.len()));
            kept.push(inbox);
        }
        if streams.is_empty() {
            return Err(WebRtcError::NothingDescribed(within).into());
        }
        if streams.len() < offered {
            pp_warn!(
                pp_log: &pp_log,
                "{} of {} tracks said what they carry within {within:?}; the rest are left out",
                streams.len(),
                offered,
            );
        }
        let identity = Identity::new(ElementType::WebRtcSource, name);
        let source = Self(SourceStage::new(
            identity,
            Watching {
                inboxes: kept,
                contracts,
                next: 0,
                _client: client,
                pp_log,
            },
        ));
        Ok((source, streams))
    }
}

impl Source for Watching {
    fn is_live(&self) -> bool {
        true
    }

    /// It waits on its inboxes, which a packet wakes.
    fn drivable(&self) -> bool {
        true
    }

    fn outputs(&self) -> Option<Vec<SrcPad>> {
        Some(
            self.contracts
                .iter()
                .enumerate()
                .map(|(index, contract)| SrcPad::with_contract(format!("src_{index}"), *contract))
                .collect(),
        )
    }

    /// The next packet of any track, the tracks taken in turn — or, once
    /// the connection has let go of every one, the end of the stream.
    fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
        let tracks = self.inboxes.len();
        loop {
            let waker = wait.waker().cloned();
            let mut ended = 0;
            for turn in 0..tracks {
                let index = (self.next + turn) % tracks;
                match self.inboxes[index].take(waker.as_ref()) {
                    Taken::Packet(buf) => {
                        self.next = (index + 1) % tracks;
                        return Ok(Produced::On(index, buf));
                    }
                    Taken::Ended => ended += 1,
                    Taken::Empty => {}
                }
            }
            if ended == tracks {
                pp_info!(pp_log: &self.pp_log, "event=eos phase=source the connection ended");
                return Ok(Produced::End);
            }
            if waker.is_some() {
                return Ok(Produced::Nothing);
            }
            match wait.rung(self.inboxes[0].bell()) {
                Received::Got(()) => {}
                Received::LetGo | Received::Gone => return Ok(Produced::Nothing),
            }
        }
    }

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
    }
}

impl Drop for Watching {
    fn drop(&mut self) {
        for inbox in &self.inboxes {
            inbox.reader_gone();
        }
    }
}
