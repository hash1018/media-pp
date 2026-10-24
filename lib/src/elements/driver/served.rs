//! What a server's path holds for as long as it is served: [`Served`].

use crate::elements::Broadcast;

/// A path's broadcast, which its muxers write into and its players read,
/// and its registration on the server: let go of together. The broadcast
/// is closed as it goes, which ends its players.
pub(crate) struct Served {
    pub(crate) broadcast: Broadcast,
    /// Takes the path's registration away as this goes.
    _taken: Box<dyn Send + Sync>,
}

impl Served {
    pub(crate) fn new(broadcast: Broadcast, taken: impl Send + Sync + 'static) -> Self {
        Self {
            broadcast,
            _taken: Box::new(taken),
        }
    }
}

impl Drop for Served {
    fn drop(&mut self) {
        self.broadcast.close();
    }
}

impl std::fmt::Debug for Served {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Served")
            .field("broadcast", &self.broadcast.name())
            .finish()
    }
}
