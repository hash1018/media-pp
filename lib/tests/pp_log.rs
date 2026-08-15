use media_pp::pp_log::{PpLog, pp_debug, pp_error, pp_info, pp_trace, pp_warn};

struct LoggedValue {
    pp_log: PpLog,
}

impl LoggedValue {
    fn emit(&self) {
        pp_info!(self, "self form");
    }
}

#[test]
fn public_pp_log_macros_accept_every_supported_target_form() {
    let pp_log = PpLog::new("element", Some("pipeline"));

    pp_info!(pp_log: &pp_log, "context form");
    pp_debug!(main_id: "main", sub_id: "sub", "main and sub form");
    pp_warn!(main_id: "main", "main form");
    pp_error!(pp_log: &pp_log, "error form");
    pp_trace!(pp_log: &pp_log, "trace form");

    LoggedValue { pp_log }.emit();
}
