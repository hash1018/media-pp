//! A publisher an [`RtmpServer`](super::RtmpServer) took, as a source of
//! what it sends: [`RtmpSource`].

use std::sync::Arc;

use crossbeam_channel::Receiver;
use ffmpeg_next as ffmpeg;

use crate::buffer::MediaBuffer;
use crate::contract::{MediaKind, OutputContract, PortContract};
use crate::core::bell::{self, Bell, Ringing};
use crate::element::{Attached, ElementType, Identity, Produced, Source, SourceStage, Wait};
use crate::elements::StreamInfo;
use crate::error::Result;
use crate::pad::SrcPad;
use crate::pp_log::{PpLog, pp_info};
use crate::produce::{Received, source_stage};

/// How many packets wait for the source at most: some seconds of a
/// publisher's pictures and sound.
pub(super) const BACKLOG: usize = 256;

/// One RTMP publisher's stream, as an [`RtmpServer`](super::RtmpServer)
/// hands it out through the [`Publishers`](crate::elements::Publishers) of
/// its path — an output for each of its tracks, the pictures before the
/// sound, as their streams say.
///
/// Nothing is decoded: H.264 and AAC as RTMP has always carried them, and
/// H.265, AV1, VP9 and Opus as Enhanced RTMP does, are handed on as they
/// come, timed in the milliseconds RTMP counts in. The publisher's
/// connection is read on the server's thread for it, and hands on through
/// a buffer of some seconds of packets; a source not reading for longer
/// than that loses what comes past it, and pictures go on from their next
/// keyframe.
///
/// It ends as the publisher does: leaving, going silent for the server's
/// `publisher_timeout`, or sending a new configuration — a new resolution
/// — which the server hands out again, as a source of its own. Dropping it
/// ends the publisher's connection.
pub struct RtmpSource(SourceStage<Taking>);

source_stage!(RtmpSource);

/// What an [`RtmpSource`] hands on, a packet at a time as the connection
/// hands it over.
pub(super) struct Taking {
    pp_log: PpLog,
    /// Each output's, by stream index.
    contracts: Vec<OutputContract>,
    rx: Receiver<(usize, ffmpeg::Packet)>,
    bell: Bell,
}

impl RtmpSource {
    /// The source of a publisher of `path` whose `streams`, numbered, its
    /// connection has said, and where the connection feeds it.
    pub(super) fn published(
        path: &str,
        streams: &[StreamInfo],
    ) -> (Self, Ringing<(usize, ffmpeg::Packet)>) {
        let identity = Identity::new(ElementType::RtmpSource, path);
        let pp_log = identity.pp_log().clone();
        let (feed, rx, bell) = bell::ringing(BACKLOG);
        let contracts = streams
            .iter()
            .map(|stream| match MediaKind::packet_for(stream.kind()) {
                Some(kind) => OutputContract::Fixed(PortContract::packet(kind)),
                None => OutputContract::Unknown,
            })
            .collect();
        let source = Self(SourceStage::new(
            identity,
            Taking {
                pp_log,
                contracts,
                rx,
                bell,
            },
        ));
        (source, feed)
    }
}

impl Source for Taking {
    fn is_live(&self) -> bool {
        true
    }

    /// One per stream, at the stream's own index.
    fn outputs(&self) -> Option<Vec<SrcPad>> {
        Some(
            self.contracts
                .iter()
                .enumerate()
                .map(|(index, contract)| SrcPad::with_contract(format!("src_{index}"), *contract))
                .collect(),
        )
    }

    /// The next packet the connection hands over, on its stream's output;
    /// the end once the connection has let go of the publisher.
    fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
        match wait.recv(&self.rx, &self.bell) {
            Received::Got((index, packet)) => Ok(Produced::On(
                index,
                MediaBuffer::Packet(Arc::new(packet).into()),
            )),
            Received::LetGo => Ok(Produced::Nothing),
            Received::Gone => {
                pp_info!(pp_log: &self.pp_log, "the publisher ended");
                Ok(Produced::End)
            }
        }
    }

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
    }
}
