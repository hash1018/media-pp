use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use media_pp::{
    buffer::MediaBuffer,
    elements::{AppSink, AppSource},
    log::{self, Level},
    pipeline::Pipeline,
};

#[test]
fn pipeline_logs_topology_eos_and_control_at_each_boundary() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after the Unix epoch")
        .as_nanos();
    let log_dir = std::env::temp_dir().join(format!(
        "media_pp_flow_log_{}_{}",
        std::process::id(),
        unique
    ));
    let log_path = log_dir.to_string_lossy();
    let guard =
        log::init("flow", &log_path, Level::Trace, 1).expect("private file logger must initialize");

    let (source, handle) = AppSource::new("source", 4);
    let pipeline = Pipeline::new("flow-test", source, |source, context| {
        let branch = context
            .branch()
            .queue("queue", 4)
            .to(Box::new(AppSink::new("sink", |_buf| Ok(()))))?;
        context.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("test pipeline must build");

    pipeline.run();
    pipeline.pause();
    pipeline.resume();
    handle
        .push(MediaBuffer::Eos)
        .expect("source must accept EOS");
    drop(handle);
    let _events: Vec<_> = pipeline.bus().iter().collect();

    assert_eq!(guard.dropped_lines(), 0);
    drop(guard);

    let log_file = fs::read_dir(&log_dir)
        .expect("temporary log directory must remain readable")
        .map(|entry| entry.expect("log directory entry must be readable").path())
        .find(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("flow."))
        })
        .expect("the rolling appender must create a log file");
    let contents = fs::read_to_string(log_file).expect("log file must contain UTF-8 text");

    assert!(contents.contains("[element=Pipeline] [name=flow-test] run"));
    assert!(contents.contains(concat!(
        "[element=Pipeline] [name=flow-test] topology\n",
        "AppSource(source)#1\n",
        "└── [source_src] → Queue(queue)#2\n",
        "                   └── [queue_src] → AppSink(sink)#3",
    )));
    assert!(
        contents.contains(
            "[element=AppSource] [name=source] event=control control=Pause phase=received"
        )
    );
    assert!(
        contents
            .contains("[element=Queue] [name=queue] event=control control=Pause phase=forwarding")
    );
    assert!(contents.contains(
        "[element=AppSink] [name=sink] event=control control=Pause phase=completed outcome=ok"
    ));
    assert!(contents.contains(
        "[element=AppSource] [name=source] event=eos phase=sent pad=source_src outcome=ok"
    ));
    assert!(contents.contains("[element=Queue] [name=queue] event=eos phase=completed outcome=ok"));
    assert!(contents.contains("[element=AppSink] [name=sink] event=eos phase=received"));

    fs::remove_dir_all(log_dir).expect("temporary log directory must be removable");
}
