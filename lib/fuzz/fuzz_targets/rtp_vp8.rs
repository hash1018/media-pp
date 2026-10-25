#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| media_pp::fuzzing::rtp_vp8(data));
