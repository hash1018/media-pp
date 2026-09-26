//! What an element allocates for the size of the frames it is handed.
//!
//! A texture, a staging buffer, an `AVHWFramesContext` pool: each is made
//! for one resolution and cannot be resized afterwards. The element that
//! owns one does not know that resolution until a frame arrives — asking
//! the caller for it instead would be asking for a value that has to agree
//! with whatever the upstream element produces, and a caller that gets it
//! wrong learns so one frame later.
//!
//! [`ForSize`] is that resource held against the size it was made for: made
//! on the first frame, and made again when a source changes resolution
//! mid-stream, which is what an RTSP camera or a window capture does. The
//! previous one is released only once its replacement is in hand, so a
//! failed allocation leaves the element exactly as it was, still able to
//! serve the size it already had.
//!
//! What this does *not* do is decide the size an element deliberately
//! chooses: a scaler's output, a compositor's canvas, an encoder's stream.
//! Those are the element's own configuration and stay constructor
//! arguments.

/// What size an element's output frames are: a size it was given, or the
/// size of whatever arrived.
///
/// The elements that resize are the ones told a size, and each of them can
/// also be asked to change only a layout — `SwScaler::to_format`,
/// `D3d11Scaler::to_format`, `CudaScaler::to_format` — which is where the
/// second case comes from. Where the size is given, a source that changes
/// resolution mid-stream is absorbed by scaling it back to it; where it is
/// the input's, that change is passed on.
#[derive(Debug, Clone, Copy)]
pub(crate) enum OutputSize {
    /// This size, whatever arrives.
    Fixed {
        /// Output width in pixels.
        width: u32,
        /// Output height in pixels.
        height: u32,
    },
    /// The size of the frame that arrived.
    OfTheInput,
}

impl OutputSize {
    /// The size `frame` is to come out as.
    pub(crate) fn of(self, frame: &ffmpeg_next::frame::Video) -> (u32, u32) {
        match self {
            Self::Fixed { width, height } => (width, height),
            Self::OfTheInput => (frame.width(), frame.height()),
        }
    }
}

/// A resource made for one frame size — see the module docs.
pub(crate) struct ForSize<T> {
    made: Option<Made<T>>,
}

struct Made<T> {
    width: u32,
    height: u32,
    value: T,
}

impl<T> ForSize<T> {
    /// Nothing made yet; the first frame decides the size.
    pub(crate) const fn new() -> Self {
        Self { made: None }
    }

    /// The resource for a `width`x`height` frame, calling `make` only when
    /// there is none yet or the one held was made for another size.
    ///
    /// For what cannot fail to allocate — a pool of CPU frames. A device
    /// allocation goes through [`ForSize::try_get`].
    pub(crate) fn get(
        &mut self,
        width: u32,
        height: u32,
        make: impl FnOnce(u32, u32) -> T,
    ) -> &mut T {
        if !self.holds(width, height) {
            self.made = Some(Made {
                width,
                height,
                value: make(width, height),
            });
        }
        self.held()
    }

    /// [`ForSize::get`] where making the resource can fail — a device
    /// allocation, so this exists for the backends that have one. The
    /// previous one is kept on failure rather than dropped: an element that
    /// cannot allocate for a new size is still the element it was.
    #[cfg(any(
        feature = "cuda",
        feature = "vulkan",
        all(target_os = "windows", any(feature = "d3d11", feature = "d3d12"))
    ))]
    pub(crate) fn try_get<E>(
        &mut self,
        width: u32,
        height: u32,
        make: impl FnOnce(u32, u32) -> std::result::Result<T, E>,
    ) -> std::result::Result<&mut T, E> {
        if !self.holds(width, height) {
            let value = make(width, height)?;
            self.made = Some(Made {
                width,
                height,
                value,
            });
        }
        Ok(self.held())
    }

    fn holds(&self, width: u32, height: u32) -> bool {
        self.made
            .as_ref()
            .is_some_and(|made| made.width == width && made.height == height)
    }

    fn held(&mut self) -> &mut T {
        &mut self
            .made
            .as_mut()
            .expect("a resource was just made for this size")
            .value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Makes a `String` for each size and records that it did.
    fn named(
        resource: &mut ForSize<String>,
        width: u32,
        height: u32,
        made: &mut Vec<(u32, u32)>,
    ) -> String {
        resource
            .get(width, height, |width, height| {
                made.push((width, height));
                format!("{width}x{height}")
            })
            .clone()
    }

    #[test]
    fn the_first_frame_decides_the_size_and_the_next_one_reuses_it() {
        let mut made = Vec::new();
        let mut resource: ForSize<String> = ForSize::new();

        for (width, height) in [(64, 32), (64, 32), (32, 16), (32, 16), (64, 32)] {
            let name = named(&mut resource, width, height, &mut made);
            assert_eq!(name, format!("{width}x{height}"));
        }

        assert_eq!(
            made,
            [(64, 32), (32, 16), (64, 32)],
            "one allocation per size change, and none while the size holds"
        );
    }

    /// An element that cannot allocate for a new size is still the element
    /// it was, serving the size it already had.
    #[cfg(any(
        feature = "cuda",
        all(target_os = "windows", any(feature = "d3d11", feature = "d3d12"))
    ))]
    #[test]
    fn a_failed_allocation_leaves_the_previous_one_in_place() {
        let mut made = Vec::new();
        let mut resource: ForSize<String> = ForSize::new();
        named(&mut resource, 64, 32, &mut made);

        let error = resource
            .try_get(32, 16, |_, _| Err::<String, _>("no memory"))
            .expect_err("the allocation failed");

        assert_eq!(error, "no memory");
        assert_eq!(
            named(&mut resource, 64, 32, &mut made),
            "64x32",
            "the size it already had is still served"
        );
        assert_eq!(made, [(64, 32)], "and was not made a second time");
    }
}
