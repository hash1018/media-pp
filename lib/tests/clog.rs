use media_pp::clog::{CLog, cdebug, cerror, cinfo, ctrace, cwarn};

struct LoggedValue {
    clog: CLog,
}

impl LoggedValue {
    fn emit(&self) {
        cinfo!(self, "self form");
    }
}

#[test]
fn public_clog_macros_accept_every_supported_target_form() {
    let clog = CLog::new("element", Some("pipeline"));

    cinfo!(clog: &clog, "context form");
    cdebug!(main_id: "main", sub_id: "sub", "main and sub form");
    cwarn!(main_id: "main", "main form");
    cerror!(clog: &clog, "error form");
    ctrace!(clog: &clog, "trace form");

    LoggedValue { clog }.emit();
}
