use env_logger::Builder;
use log::LevelFilter;
use rust_hlog::{HLog, hdebug, herror, hinfo, hlog, htrace, hwarn};

#[hlog]
struct LogTest {}

impl LogTest {
    fn new() -> Self {
        LogTest {
            hlog: HLog::new("LogTest", Some("SubID")),
        }
    }

    fn test(&self) {
        let hlog = HLog::new("LogTest2", Some("SubID"));
        hinfo!(self, "Hello, World!");
        hdebug!(self, "Hello, World!");
        hwarn!(self, "Hello, World!");
        herror!(self, "Hello, World!");
        htrace!(self, "Hello, World!");
        hinfo!(main_id:"main_id", sub_id:"sub_id", "Hello, World!");
        hdebug!(main_id:"main_id", sub_id:"sub_id", "Hello, World!");
        hwarn!(main_id:"main_id", sub_id:"sub_id", "Hello, World!");
        herror!(main_id:"main_id", sub_id:"sub_id", "Hello, World!");
        htrace!(main_id:"main_id", sub_id:"sub_id", "Hello, World!");

        hinfo!(hlog:hlog, "Hello, World!");
        hdebug!(hlog:hlog, "Hello, World!");
        hwarn!(hlog:hlog, "Hello, World!");
        herror!(hlog:hlog, "Hello, World!");
        htrace!(hlog:hlog, "Hello, World!");
    }
}
fn main() {
    Builder::new().filter_level(LevelFilter::max()).init();

    let log_test = LogTest::new();
    log_test.test();
}
