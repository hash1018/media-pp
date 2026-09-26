//! What every video compositor's handles do, whichever backend draws.
//!
//! Each compositor has handles of its own — [`SwVideoCompositorHandle`],
//! `D3d11VideoCompositorHandle`, `CudaVideoCompositorHandle`,
//! `VulkanVideoCompositorHandle` — with the same methods, each answering its own error type. These traits are those
//! methods once, answering the crate's [`Error`], so code that builds a
//! composition — a Scene, an editing timeline — is written once and handed
//! whichever compositor the platform has.
//!
//! A backend's own handle is still what to reach for where the backend is
//! known: it says exactly which errors it can give, and has what only that
//! backend has.
//!
//! [`SwVideoCompositorHandle`]: crate::elements::SwVideoCompositorHandle

use std::sync::Arc;

use ffmpeg_next as ffmpeg;

use super::text_layer::TextLayer;
use super::video_layer::{VideoFit, VideoInputId, VideoLayer, VideoRect, VideoSourceRect};
use crate::{element::Sink, error::Error, pool::UnboundObjectPoolRef};

type Result<T> = std::result::Result<T, Error>;

/// A picture as a compositor holds one: the pooled frame, shared.
pub type LayerFrame = Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>;

/// The two ends of one input fed through a pipeline: the sink that ends
/// that pipeline, and the handle that places what it feeds.
///
/// Move `sink` into the input's own pipeline and keep `layer` for moving,
/// showing and restacking it while it runs.
pub struct CompositorInput<L> {
    /// The terminal sink to end the input's pipeline with.
    pub sink: Box<dyn Sink>,
    /// Runtime control of where and how it is drawn.
    pub layer: L,
}

/// Adding, removing and pacing a video compositor's inputs.
///
/// Implemented by every compositor's handle: see the module docs. What each
/// method does is what the backend's own method of the same name does.
pub trait VideoCompositorControl: Clone + Send + 'static {
    /// The handle each video input is placed through.
    type Layer: VideoLayerControl;
    /// The handle each text layer is written and placed through.
    type Text: TextLayerControl;

    /// Registers an input fed through a pipeline, replacing any of the
    /// same name.
    fn add_source(
        &self,
        name: impl Into<String>,
        layer: VideoLayer,
    ) -> Result<CompositorInput<Self::Layer>>;

    /// Registers an input whose picture is set through its handle —
    /// [`VideoLayerControl::set_frame`] — rather than a pipeline, replacing
    /// any of the same name.
    fn add_layer(&self, name: impl Into<String>, layer: VideoLayer) -> Result<Self::Layer>;

    /// Registers a text layer, replacing any of the same name.
    fn add_text_layer(&self, name: impl Into<String>, text: TextLayer) -> Result<Self::Text>;

    /// Removes whatever is registered under `name`.
    fn remove_source(&self, name: &str);

    /// How many inputs are registered.
    fn source_count(&self) -> usize;

    /// The rate the compositor emits at, or `None` once it is gone.
    fn frame_rate(&self) -> Option<ffmpeg::Rational>;

    /// Changes the rate the compositor emits at.
    fn set_frame_rate(&self, frame_rate: ffmpeg::Rational) -> Result<()>;
}

/// Placing one video input: where it is drawn, how, and — for an input
/// added with [`VideoCompositorControl::add_layer`] — what.
pub trait VideoLayerControl: Clone + Send + 'static {
    /// The identity of this registration, which a same-name replacement
    /// does not share.
    fn id(&self) -> VideoInputId;
    /// The name it was registered under.
    fn name(&self) -> Arc<str>;
    /// Its settings now, or `None` once it has been removed.
    fn layer(&self) -> Option<VideoLayer>;
    /// The frame it will draw next, if it has one.
    fn latest_frame(&self) -> Option<LayerFrame>;
    /// Replaces every setting at once.
    fn set_layer(&self, layer: VideoLayer) -> Result<()>;
    /// Moves and sizes it.
    fn set_rect(&self, rect: VideoRect) -> Result<()>;
    /// Blends it at `opacity`, from 0 to 1.
    fn set_opacity(&self, opacity: f32) -> Result<()>;
    /// Restacks it: a higher `z_index` is drawn over a lower one.
    fn set_z_index(&self, z_index: i32) -> Result<()>;
    /// Shows or hides it.
    fn set_visible(&self, visible: bool) -> Result<()>;
    /// How its picture fits its rectangle.
    fn set_fit(&self, fit: VideoFit) -> Result<()>;
    /// The part of its picture drawn, or all of it for `None`.
    fn set_source(&self, source: Option<VideoSourceRect>) -> Result<()>;
    /// Makes `frame` its picture, until another replaces it. The frame has
    /// to be one this backend draws from, as one through a sink does.
    fn set_frame(&self, frame: LayerFrame) -> Result<()>;
}

/// Writing and placing one text layer.
pub trait TextLayerControl: Send + 'static {
    /// What it says; empty text draws nothing.
    fn set_text(&self, text: &str) -> Result<()>;
    /// Where its top-left corner is, on the canvas.
    fn set_position(&self, x: i32, y: i32) -> Result<()>;
    /// Blends it at `opacity`, from 0 to 1.
    fn set_opacity(&self, opacity: f32) -> Result<()>;
    /// Restacks it among the video layers and other text layers.
    fn set_z_index(&self, z_index: i32) -> Result<()>;
    /// Shows or hides it.
    fn set_visible(&self, visible: bool) -> Result<()>;
}

/// Implements the three traits for one backend by delegating to the
/// methods of the same names on its own handles — which is what they are.
macro_rules! compositor_control {
    ($handle:ty, $layer:ty, $text:ty) => {
        impl $crate::elements::VideoCompositorControl for $handle {
            type Layer = $layer;
            type Text = $text;

            fn add_source(
                &self,
                name: impl Into<String>,
                layer: $crate::elements::VideoLayer,
            ) -> ::std::result::Result<
                $crate::elements::CompositorInput<$layer>,
                $crate::error::Error,
            > {
                Ok(<$handle>::add_source(self, name, layer)?)
            }

            fn add_layer(
                &self,
                name: impl Into<String>,
                layer: $crate::elements::VideoLayer,
            ) -> ::std::result::Result<$layer, $crate::error::Error> {
                Ok(<$handle>::add_layer(self, name, layer)?)
            }

            fn add_text_layer(
                &self,
                name: impl Into<String>,
                text: $crate::elements::TextLayer,
            ) -> ::std::result::Result<$text, $crate::error::Error> {
                Ok(<$handle>::add_text_layer(self, name, text)?)
            }

            fn remove_source(&self, name: &str) {
                <$handle>::remove_source(self, name)
            }

            fn source_count(&self) -> usize {
                <$handle>::source_count(self)
            }

            fn frame_rate(&self) -> Option<::ffmpeg_next::Rational> {
                <$handle>::frame_rate(self)
            }

            fn set_frame_rate(
                &self,
                frame_rate: ::ffmpeg_next::Rational,
            ) -> ::std::result::Result<(), $crate::error::Error> {
                Ok(<$handle>::set_frame_rate(self, frame_rate)?)
            }
        }

        impl $crate::elements::VideoLayerControl for $layer {
            fn id(&self) -> $crate::elements::VideoInputId {
                <$layer>::id(self)
            }

            fn name(&self) -> ::std::sync::Arc<str> {
                <$layer>::name(self)
            }

            fn layer(&self) -> Option<$crate::elements::VideoLayer> {
                <$layer>::layer(self)
            }

            fn latest_frame(&self) -> Option<$crate::elements::LayerFrame> {
                <$layer>::latest_frame(self)
            }

            fn set_layer(
                &self,
                layer: $crate::elements::VideoLayer,
            ) -> ::std::result::Result<(), $crate::error::Error> {
                Ok(<$layer>::set_layer(self, layer)?)
            }

            fn set_rect(
                &self,
                rect: $crate::elements::VideoRect,
            ) -> ::std::result::Result<(), $crate::error::Error> {
                Ok(<$layer>::set_rect(self, rect)?)
            }

            fn set_opacity(&self, opacity: f32) -> ::std::result::Result<(), $crate::error::Error> {
                Ok(<$layer>::set_opacity(self, opacity)?)
            }

            fn set_z_index(&self, z_index: i32) -> ::std::result::Result<(), $crate::error::Error> {
                Ok(<$layer>::set_z_index(self, z_index)?)
            }

            fn set_visible(
                &self,
                visible: bool,
            ) -> ::std::result::Result<(), $crate::error::Error> {
                Ok(<$layer>::set_visible(self, visible)?)
            }

            fn set_fit(
                &self,
                fit: $crate::elements::VideoFit,
            ) -> ::std::result::Result<(), $crate::error::Error> {
                Ok(<$layer>::set_fit(self, fit)?)
            }

            fn set_source(
                &self,
                source: Option<$crate::elements::VideoSourceRect>,
            ) -> ::std::result::Result<(), $crate::error::Error> {
                Ok(<$layer>::set_source(self, source)?)
            }

            fn set_frame(
                &self,
                frame: $crate::elements::LayerFrame,
            ) -> ::std::result::Result<(), $crate::error::Error> {
                Ok(<$layer>::set_frame(self, frame)?)
            }
        }
    };
}

pub(crate) use compositor_control;

/// Code written against these traits alone, run by each backend's tests:
/// an input fed through a sink, one set through its handle and raised over
/// it, and the first removed again. Returns the second, for the backend to
/// give a picture of its own.
#[cfg(test)]
pub(crate) fn arrange<C: VideoCompositorControl>(handle: &C, width: u32, height: u32) -> C::Layer {
    let fed = handle
        .add_source("fed", VideoLayer::new(VideoRect::new(0, 0, width, height)))
        .expect("add an input fed through a sink");
    let still = handle
        .add_layer(
            "still",
            VideoLayer::new(VideoRect::new(0, 0, width, height)),
        )
        .expect("add an input set through its handle");
    still.set_z_index(1).expect("restack it");
    assert_eq!(handle.source_count(), 2);
    assert_eq!(still.layer().map(|layer| layer.z_index), Some(1));
    assert_eq!(&*still.name(), "still");
    drop(fed);
    handle.remove_source("fed");
    assert_eq!(handle.source_count(), 1);
    still
}
