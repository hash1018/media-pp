//! Contextual logging identity and macros used by pipeline elements.
//!
//! A [`PpLog`] stores the stable log target for one element. The `pp_*`
//! macros route messages through the `log` facade while attaching that
//! target, so the configured subscriber can attribute each record to the
//! exact element and pipeline that produced it.

/// Stable contextual identity attached to a pipeline element's log records.
#[derive(Debug, Clone)]
pub struct PpLog {
    log_id: String,
}

impl PpLog {
    pub fn new(id: &str, sub_id: Option<&str>) -> Self {
        Self {
            log_id: make_log_id(id, sub_id),
        }
    }

    pub fn set_log_id(&mut self, id: &str, sub_id: Option<&str>) {
        self.log_id = make_log_id(id, sub_id);
    }

    pub fn log_id(&self) -> &str {
        &self.log_id
    }
}

fn make_log_id(id: &str, sub_id: Option<&str>) -> String {
    match sub_id {
        Some(sub_id) => format!("{id}:{sub_id}"),
        None => id.to_owned(),
    }
}

#[doc(hidden)]
pub mod __private {
    pub use log::{Level, log};
}

#[macro_export]
macro_rules! pp_info {
    (pp_log: $pp_log:expr, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: $pp_log.log_id(),
            $crate::pp_log::__private::Level::Info,
            $($arg)+
        )
    };
    ($self:ident, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: $self.pp_log.log_id(),
            $crate::pp_log::__private::Level::Info,
            $($arg)+
        )
    };
    (main_id: $main_id:expr, sub_id: $sub_id:expr, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: &format!("{}:{}", $main_id, $sub_id),
            $crate::pp_log::__private::Level::Info,
            $($arg)+
        )
    };
    (main_id: $main_id:expr, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: $main_id,
            $crate::pp_log::__private::Level::Info,
            $($arg)+
        )
    };
}

#[macro_export]
macro_rules! pp_debug {
    (pp_log: $pp_log:expr, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: $pp_log.log_id(),
            $crate::pp_log::__private::Level::Debug,
            $($arg)+
        )
    };
    ($self:ident, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: $self.pp_log.log_id(),
            $crate::pp_log::__private::Level::Debug,
            $($arg)+
        )
    };
    (main_id: $main_id:expr, sub_id: $sub_id:expr, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: &format!("{}:{}", $main_id, $sub_id),
            $crate::pp_log::__private::Level::Debug,
            $($arg)+
        )
    };
    (main_id: $main_id:expr, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: $main_id,
            $crate::pp_log::__private::Level::Debug,
            $($arg)+
        )
    };
}

#[macro_export]
macro_rules! pp_warn {
    (pp_log: $pp_log:expr, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: $pp_log.log_id(),
            $crate::pp_log::__private::Level::Warn,
            $($arg)+
        )
    };
    ($self:ident, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: $self.pp_log.log_id(),
            $crate::pp_log::__private::Level::Warn,
            $($arg)+
        )
    };
    (main_id: $main_id:expr, sub_id: $sub_id:expr, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: &format!("{}:{}", $main_id, $sub_id),
            $crate::pp_log::__private::Level::Warn,
            $($arg)+
        )
    };
    (main_id: $main_id:expr, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: $main_id,
            $crate::pp_log::__private::Level::Warn,
            $($arg)+
        )
    };
}

#[macro_export]
macro_rules! pp_error {
    (pp_log: $pp_log:expr, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: $pp_log.log_id(),
            $crate::pp_log::__private::Level::Error,
            $($arg)+
        )
    };
    ($self:ident, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: $self.pp_log.log_id(),
            $crate::pp_log::__private::Level::Error,
            $($arg)+
        )
    };
    (main_id: $main_id:expr, sub_id: $sub_id:expr, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: &format!("{}:{}", $main_id, $sub_id),
            $crate::pp_log::__private::Level::Error,
            $($arg)+
        )
    };
    (main_id: $main_id:expr, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: $main_id,
            $crate::pp_log::__private::Level::Error,
            $($arg)+
        )
    };
}

#[macro_export]
macro_rules! pp_trace {
    (pp_log: $pp_log:expr, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: $pp_log.log_id(),
            $crate::pp_log::__private::Level::Trace,
            $($arg)+
        )
    };
    ($self:ident, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: $self.pp_log.log_id(),
            $crate::pp_log::__private::Level::Trace,
            $($arg)+
        )
    };
    (main_id: $main_id:expr, sub_id: $sub_id:expr, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: &format!("{}:{}", $main_id, $sub_id),
            $crate::pp_log::__private::Level::Trace,
            $($arg)+
        )
    };
    (main_id: $main_id:expr, $($arg:tt)+) => {
        $crate::pp_log::__private::log!(
            target: $main_id,
            $crate::pp_log::__private::Level::Trace,
            $($arg)+
        )
    };
}

pub use crate::{pp_debug, pp_error, pp_info, pp_trace, pp_warn};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combines_main_and_sub_ids() {
        let pp_log = PpLog::new("element", Some("pipeline"));
        assert_eq!(pp_log.log_id(), "element:pipeline");
    }

    #[test]
    fn replaces_the_log_id() {
        let mut pp_log = PpLog::new("old", None);
        pp_log.set_log_id("new", Some("sub"));
        assert_eq!(pp_log.log_id(), "new:sub");
    }
}
