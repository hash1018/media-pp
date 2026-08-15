//! Contextual logging identity and macros used by pipeline elements.
//!
//! A [`PpLog`] stores the stable, structured identity for one element. The `pp_*`
//! macros send records only to [`crate::log`]'s opt-in private file writer,
//! keeping them isolated from an embedding application's global logger.

use std::sync::Arc;

/// Stable contextual identity attached to a pipeline element's log records.
#[derive(Debug, Clone)]
pub struct PpLog {
    pipeline_id: Option<Arc<str>>,
    element: Arc<str>,
    name: Arc<str>,
}

impl PpLog {
    /// Creates a logging identity.
    ///
    /// `element` is the element type (for example `FileDemuxer`), while
    /// `name` is the caller-selected instance name (for example `demux`).
    /// `pipeline_id` is omitted until the element is attached to a pipeline.
    pub fn new(element: &str, name: &str, pipeline_id: Option<&str>) -> Self {
        Self {
            pipeline_id: pipeline_id.map(Arc::from),
            element: Arc::from(element),
            name: Arc::from(name),
        }
    }

    pub fn pipeline_id(&self) -> Option<&str> {
        self.pipeline_id.as_deref()
    }

    pub fn element(&self) -> &str {
        &self.element
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[doc(hidden)]
pub mod __private {
    pub use crate::log::{Level, emit, enabled};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __pp_log {
    ($level:expr, pp_log: $pp_log:expr, $($arg:tt)+) => {{
        let level = $level;
        if $crate::pp_log::__private::enabled(level) {
            $crate::pp_log::__private::emit(
                level,
                $pp_log,
                format_args!($($arg)+),
            );
        }
    }};
    ($level:expr, $self:ident, $($arg:tt)+) => {{
        let level = $level;
        if $crate::pp_log::__private::enabled(level) {
            $crate::pp_log::__private::emit(
                level,
                &$self.pp_log,
                format_args!($($arg)+),
            );
        }
    }};
}

#[macro_export]
macro_rules! pp_info {
    ($($arg:tt)+) => {
        $crate::__pp_log!($crate::pp_log::__private::Level::Info, $($arg)+)
    };
}

#[macro_export]
macro_rules! pp_debug {
    ($($arg:tt)+) => {
        $crate::__pp_log!($crate::pp_log::__private::Level::Debug, $($arg)+)
    };
}

#[macro_export]
macro_rules! pp_warn {
    ($($arg:tt)+) => {
        $crate::__pp_log!($crate::pp_log::__private::Level::Warn, $($arg)+)
    };
}

#[macro_export]
macro_rules! pp_error {
    ($($arg:tt)+) => {
        $crate::__pp_log!($crate::pp_log::__private::Level::Error, $($arg)+)
    };
}

#[macro_export]
macro_rules! pp_trace {
    ($($arg:tt)+) => {
        $crate::__pp_log!($crate::pp_log::__private::Level::Trace, $($arg)+)
    };
}

pub use crate::{pp_debug, pp_error, pp_info, pp_trace, pp_warn};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_each_identity_field_separate() {
        let pp_log = PpLog::new("FileDemuxer", "demux", Some("app-sink"));
        assert_eq!(pp_log.pipeline_id(), Some("app-sink"));
        assert_eq!(pp_log.element(), "FileDemuxer");
        assert_eq!(pp_log.name(), "demux");
    }
}
