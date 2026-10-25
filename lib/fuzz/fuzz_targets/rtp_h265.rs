#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| media_pp::fuzzing::rtp_h265(data));
