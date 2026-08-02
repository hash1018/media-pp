extern crate hlog_macro;
pub use hlog_macro::hlog;
pub use log::Level;
pub use log::log;

#[derive(Debug, Clone)]
pub struct HLog {
    log_id: String,
}

impl HLog {
    pub fn new(id: &str, sub_id: Option<&str>) -> Self {
        let log_id = make_log_id(id, sub_id);
        HLog { log_id }
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
        Some(sub_id) => format!("{}:{}", id, sub_id),
        None => id.to_string(),
    }
}

#[macro_export]
macro_rules! hinfo {
    (hlog: $hlog:expr, $($arg:tt)+) => ($crate::log!(target: $hlog.log_id(), $crate::Level::Info, $($arg)+));

    ($self:ident, $($arg:tt)+) => ($crate::log!(target:$self.hlog.log_id(), $crate::Level::Info, $($arg)+));

    (main_id:$main_id:expr, sub_id:$sub_id:expr, $($arg:tt)+) => ($crate::log!(target:&format!("{}:{}", $main_id, $sub_id), $crate::Level::Info, $($arg)+));
    (main_id:$main_id:expr, $($arg:tt)+) => ($crate::log!(target:$main_id, $crate::Level::Info, $($arg)+));
}

#[macro_export]
macro_rules! hdebug {
    (hlog: $hlog:expr, $($arg:tt)+) => ($crate::log!(target: $hlog.log_id(), $crate::Level::Debug, $($arg)+));

    ($self:ident, $($arg:tt)+) => ($crate::log!(target:$self.hlog.log_id(), $crate::Level::Debug, $($arg)+));

    (main_id:$main_id:expr, sub_id:$sub_id:expr, $($arg:tt)+) => ($crate::log!(target:&format!("{}:{}", $main_id, $sub_id), $crate::Level::Debug, $($arg)+));
    (main_id:$main_id:expr, $($arg:tt)+) => ($crate::log!(target:$main_id, $crate::Level::Debug, $($arg)+));
}

#[macro_export]
macro_rules! hwarn {
    (hlog: $hlog:expr, $($arg:tt)+) => ($crate::log!(target: $hlog.log_id(), $crate::Level::Warn, $($arg)+));

    ($self:ident, $($arg:tt)+) => ($crate::log!(target:$self.hlog.log_id(), $crate::Level::Warn, $($arg)+));

    (main_id:$main_id:expr, sub_id:$sub_id:expr, $($arg:tt)+) => ($crate::log!(target:&format!("{}:{}", $main_id, $sub_id), $crate::Level::Warn, $($arg)+));
    (main_id:$main_id:expr, $($arg:tt)+) => ($crate::log!(target:$main_id, $crate::Level::Warn, $($arg)+));
}

#[macro_export]
macro_rules! herror {
    (hlog: $hlog:expr, $($arg:tt)+) => ($crate::log!(target: $hlog.log_id(), $crate::Level::Error, $($arg)+));

    ($self:ident, $($arg:tt)+) => ($crate::log!(target:$self.hlog.log_id(), $crate::Level::Error, $($arg)+));

    (main_id:$main_id:expr, sub_id:$sub_id:expr, $($arg:tt)+) => ($crate::log!(target:&format!("{}:{}", $main_id, $sub_id), $crate::Level::Error, $($arg)+));
    (main_id:$main_id:expr, $($arg:tt)+) => ($crate::log!(target:$main_id, $crate::Level::Error, $($arg)+));
}

#[macro_export]
macro_rules! htrace {
    (hlog: $hlog:expr, $($arg:tt)+) => ($crate::log!(target: $hlog.log_id(), $crate::Level::Trace, $($arg)+));

    ($self:ident, $($arg:tt)+) => ($crate::log!(target:$self.hlog.log_id(), $crate::Level::Trace, $($arg)+));

    (main_id:$main_id:expr, sub_id:$sub_id:expr, $($arg:tt)+) => ($crate::log!(target:&format!("{}:{}", $main_id, $sub_id), $crate::Level::Trace, $($arg)+));
    (main_id:$main_id:expr, $($arg:tt)+) => ($crate::log!(target:$main_id, $crate::Level::Trace, $($arg)+));
}
