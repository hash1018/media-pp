//! A load test for an RTSP server: as many players as asked, each a thread
//! that plays the stream and counts what it is sent — packets, bytes, the
//! ones lost, and how steadily its pictures arrive — and a summary at the
//! end. `docs/benchmarks/streaming` is made with it, against the
//! `rtsp_server` example.
//!
//! The players are written for this, with nothing decoded, so that what is
//! measured is the server: a player is a socket and a loop that reads RTP
//! out of what the socket gives it, not FFmpeg. Each asks for every track
//! the server describes, interleaved on its RTSP connection or — with
//! `--udp` — on two UDP ports of its own per track.
//!
//! How steadily pictures arrive is a picture's arrival less the newest
//! timestamp sent yet — pictures sent in decoding order go back and forth in
//! time — taken for each picture's last packet: its spread — the largest less the
//! smallest — over the run after its first three seconds, is how far behind
//! the stream a player fell at worst. A server keeping up holds it near the
//! stream's own jitter; one falling behind lets it grow until the player is
//! moved on to a newer keyframe, which shows as packets lost.
//!
//!     cargo run --release -p rtsp_load -- rtsp://127.0.0.1:8554/file 100 [--seconds 30] [--udp]

fn main() -> Result<(), media_pp::Report> {
    Ok(example::run()?)
}

mod example {
    use std::io::{Read, Write};
    use std::net::{TcpStream, UdpSocket};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    const USAGE: &str = "usage: rtsp_load <rtsp://host:port/path> <players> [--seconds N] [--udp]";

    /// How long after it starts a player's pictures count towards its
    /// spread: past the burst of the group of pictures it joins with.
    const WARM_UP: Duration = Duration::from_secs(3);

    struct Args {
        url: String,
        players: usize,
        seconds: u64,
        udp: bool,
    }

    fn args() -> Option<Args> {
        let mut positional = Vec::new();
        let mut seconds = 30;
        let mut udp = false;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--seconds" => seconds = args.next()?.parse().ok()?,
                "--udp" => udp = true,
                _ => positional.push(arg),
            }
        }
        let [url, players] = <[String; 2]>::try_from(positional).ok()?;
        Some(Args {
            url,
            players: players.parse().ok().filter(|&n| n > 0)?,
            seconds,
            udp,
        })
    }

    /// What every player has been sent so far, together — for the reports
    /// while it runs.
    #[derive(Default)]
    struct Totals {
        bytes: AtomicU64,
        playing: AtomicU64,
    }

    /// What one player saw.
    #[derive(Default, Clone)]
    struct Seen {
        packets: u64,
        bytes: u64,
        lost: u64,
        pictures: u64,
        /// The smallest and largest arrival-less-timestamp after the warm-up,
        /// in seconds.
        lag: Option<(f64, f64)>,
        failed: Option<String>,
    }

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let Some(args) = args() else {
            eprintln!("{USAGE}");
            std::process::exit(1);
        };
        let stop = Arc::new(AtomicBool::new(false));
        let totals = Arc::new(Totals::default());
        println!(
            "{} players of {} over {} for {} s",
            args.players,
            args.url,
            if args.udp { "UDP" } else { "TCP" },
            args.seconds
        );
        let began = Instant::now();
        let players: Vec<_> = (0..args.players)
            .map(|n| {
                let (url, stop, totals, udp) = (
                    args.url.clone(),
                    Arc::clone(&stop),
                    Arc::clone(&totals),
                    args.udp,
                );
                // A few milliseconds apart, as players arrive.
                thread::sleep(Duration::from_millis(5));
                thread::Builder::new()
                    .name(format!("player-{n}"))
                    .stack_size(256 * 1024)
                    .spawn(move || play(&url, udp, &stop, &totals))
                    .expect("a thread for a player")
            })
            .collect();

        let end = began + Duration::from_secs(args.seconds);
        let mut last = (Instant::now(), 0u64);
        while Instant::now() < end {
            thread::sleep(
                Duration::from_secs(5).min(end.saturating_duration_since(Instant::now())),
            );
            let bytes = totals.bytes.load(Ordering::Relaxed);
            let elapsed = last.0.elapsed().as_secs_f64().max(1e-3);
            println!(
                "{:>4.0} s: {} playing, {:.1} Mbit/s",
                began.elapsed().as_secs_f64(),
                totals.playing.load(Ordering::Relaxed),
                (bytes - last.1) as f64 * 8.0 / elapsed / 1e6
            );
            last = (Instant::now(), bytes);
        }
        stop.store(true, Ordering::Relaxed);
        let seen: Vec<Seen> = players
            .into_iter()
            .map(|player| {
                player.join().unwrap_or_else(|_| Seen {
                    failed: Some("its thread panicked".into()),
                    ..Seen::default()
                })
            })
            .collect();
        summarize(&seen, args.seconds);
        Ok(())
    }

    fn summarize(seen: &[Seen], seconds: u64) {
        let failed: Vec<&Seen> = seen.iter().filter(|s| s.failed.is_some()).collect();
        let played: Vec<&Seen> = seen.iter().filter(|s| s.failed.is_none()).collect();
        println!("players: {} played, {} failed", played.len(), failed.len());
        for (n, failure) in failed.iter().take(5).enumerate() {
            println!("  failure {n}: {}", failure.failed.as_deref().unwrap_or(""));
        }
        if played.is_empty() {
            return;
        }
        let mbit: Vec<f64> = played
            .iter()
            .map(|s| s.bytes as f64 * 8.0 / seconds as f64 / 1e6)
            .collect();
        let total: f64 = mbit.iter().sum();
        let lost: u64 = played.iter().map(|s| s.lost).sum();
        let packets: u64 = played.iter().map(|s| s.packets).sum();
        let with_loss = played.iter().filter(|s| s.lost > 0).count();
        let mut spreads: Vec<f64> = played
            .iter()
            .filter_map(|s| s.lag.map(|(min, max)| (max - min) * 1e3))
            .collect();
        spreads.sort_by(f64::total_cmp);
        let at = |q: f64| {
            spreads
                .get(((spreads.len() as f64 - 1.0) * q).round() as usize)
                .copied()
                .unwrap_or(f64::NAN)
        };
        println!(
            "received: {total:.1} Mbit/s in all, {:.2} to {:.2} per player",
            mbit.iter().copied().fold(f64::INFINITY, f64::min),
            mbit.iter().copied().fold(0.0, f64::max)
        );
        println!(
            "pictures: {} in all; packets lost: {lost} of {} ({:.3}%), {with_loss} players lost any",
            played.iter().map(|s| s.pictures).sum::<u64>(),
            packets + lost,
            lost as f64 * 100.0 / (packets + lost).max(1) as f64
        );
        println!(
            "lag spread: median {:.1} ms, p99 {:.1} ms, max {:.1} ms",
            at(0.5),
            at(0.99),
            at(1.0)
        );
    }

    /// One player, from its first request until it is told to stop.
    fn play(url: &str, udp: bool, stop: &AtomicBool, totals: &Totals) -> Seen {
        let mut seen = Seen::default();
        if let Err(error) = session(url, udp, stop, totals, &mut seen) {
            seen.failed = Some(error);
        }
        seen
    }

    /// One track as a player follows it.
    struct Track {
        clock: f64,
        video: bool,
        expected: Option<u16>,
        /// The last timestamp, and where it has been unwrapped to.
        timestamp: Option<(u32, i64)>,
        /// The newest unwrapped timestamp yet: pictures sent in decoding
        /// order go back and forth in time, and a picture is late only by
        /// the newest one sent before it.
        newest: i64,
    }

    fn session(
        url: &str,
        udp: bool,
        stop: &AtomicBool,
        totals: &Totals,
        seen: &mut Seen,
    ) -> Result<(), String> {
        let authority = url
            .strip_prefix("rtsp://")
            .and_then(|rest| rest.split('/').next())
            .ok_or("not an rtsp:// URL")?;
        let mut connection = TcpStream::connect(authority).map_err(|e| format!("connect: {e}"))?;
        connection
            .set_read_timeout(Some(Duration::from_secs(10)))
            .map_err(|e| e.to_string())?;
        let mut cseq = 0;
        let mut pending = Vec::new();
        let described = request(
            &mut connection,
            &mut pending,
            &mut cseq,
            "DESCRIBE",
            url,
            &[],
        )?;
        let (base, sections) = sections(url, &described)?;
        let mut tracks = Vec::new();
        let mut sockets = Vec::new();
        // Each track's RTCP port is held for as long as it plays, though
        // nothing is read from it: let go, another player could be given it
        // for its RTP, and the server's sender reports to this one would
        // arrive there and read as a jump in its sequence numbers.
        let mut rtcp_ports = Vec::new();
        let mut session_id: Option<String> = None;
        for (index, (control, clock, video)) in sections.into_iter().enumerate() {
            let transport = if udp {
                let rtp = UdpSocket::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
                let rtcp = UdpSocket::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
                let ports = (
                    rtp.local_addr().map_err(|e| e.to_string())?.port(),
                    rtcp.local_addr().map_err(|e| e.to_string())?.port(),
                );
                receive_buffer(&rtp, 8 * 1024 * 1024);
                sockets.push(rtp);
                rtcp_ports.push(rtcp);
                format!("RTP/AVP;unicast;client_port={}-{}", ports.0, ports.1)
            } else {
                format!(
                    "RTP/AVP/TCP;unicast;interleaved={}-{}",
                    index * 2,
                    index * 2 + 1
                )
            };
            let target = if control.starts_with("rtsp://") {
                control
            } else {
                format!("{base}{control}")
            };
            let mut headers = vec![("Transport", transport)];
            if let Some(id) = &session_id {
                headers.push(("Session", id.clone()));
            }
            let response = request(
                &mut connection,
                &mut pending,
                &mut cseq,
                "SETUP",
                &target,
                &headers,
            )?;
            if session_id.is_none() {
                session_id = header(&response, "Session")
                    .map(|value| value.split(';').next().unwrap_or("").trim().to_owned());
            }
            tracks.push(Track {
                clock,
                video,
                expected: None,
                timestamp: None,
                newest: 0,
            });
        }
        let id = session_id.ok_or("no session")?;
        request(
            &mut connection,
            &mut pending,
            &mut cseq,
            "PLAY",
            url,
            &[("Session", id.clone())],
        )?;
        totals.playing.fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        let result = if udp {
            receive_udp(&sockets, &mut tracks, started, stop, totals, seen)
        } else {
            receive_tcp(
                &mut connection,
                pending,
                &mut tracks,
                started,
                stop,
                totals,
                seen,
            )
        };
        totals.playing.fetch_sub(1, Ordering::Relaxed);
        let _ = connection.write_all(
            format!(
                "TEARDOWN {url} RTSP/1.0\r\nCSeq: {}\r\nSession: {id}\r\n\r\n",
                cseq + 1
            )
            .as_bytes(),
        );
        result
    }

    fn receive_tcp(
        connection: &mut TcpStream,
        mut buffer: Vec<u8>,
        tracks: &mut [Track],
        started: Instant,
        stop: &AtomicBool,
        totals: &Totals,
        seen: &mut Seen,
    ) -> Result<(), String> {
        // Not with a read timeout, which on Windows loses what arrives as it
        // runs out — a write's worth of the stream now and then, counted as
        // the server's loss: read without waiting, as `receive_udp` does.
        connection
            .set_nonblocking(true)
            .map_err(|e| e.to_string())?;
        let mut chunk = vec![0u8; 256 * 1024];
        while !stop.load(Ordering::Relaxed) {
            let read = match connection.read(&mut chunk) {
                Ok(0) => return Err("the server closed the connection".into()),
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(2));
                    continue;
                }
                Err(e) => return Err(format!("read: {e}")),
            };
            buffer.extend_from_slice(&chunk[..read]);
            let now = started.elapsed();
            let mut at = 0;
            while buffer.len() >= at + 4 {
                if buffer[at] != b'$' {
                    // A response to nothing it asked, or a stray byte: past it.
                    at += 1;
                    continue;
                }
                let length = usize::from(u16::from_be_bytes([buffer[at + 2], buffer[at + 3]]));
                if buffer.len() < at + 4 + length {
                    break;
                }
                let channel = usize::from(buffer[at + 1]);
                if channel % 2 == 0
                    && let Some(track) = tracks.get_mut(channel / 2)
                {
                    rtp(&buffer[at + 4..at + 4 + length], track, now, totals, seen);
                }
                at += 4 + length;
            }
            buffer.drain(..at);
        }
        connection
            .set_nonblocking(false)
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn receive_udp(
        sockets: &[UdpSocket],
        tracks: &mut [Track],
        started: Instant,
        stop: &AtomicBool,
        totals: &Totals,
        seen: &mut Seen,
    ) -> Result<(), String> {
        let mut datagram = vec![0u8; 65_536];
        for socket in sockets {
            socket.set_nonblocking(true).map_err(|e| e.to_string())?;
        }
        while !stop.load(Ordering::Relaxed) {
            let mut got = false;
            for (index, socket) in sockets.iter().enumerate() {
                while let Ok(length) = socket.recv(&mut datagram) {
                    got = true;
                    rtp(
                        &datagram[..length],
                        &mut tracks[index],
                        started.elapsed(),
                        totals,
                        seen,
                    );
                }
            }
            if !got {
                thread::sleep(Duration::from_millis(2));
            }
        }
        Ok(())
    }

    /// Asks for a receive buffer of `bytes`: a keyframe larger than the
    /// system's default comes all at once, and would overflow it. Without
    /// a crate for it, the call itself.
    fn receive_buffer(socket: &UdpSocket, bytes: i32) {
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawSocket;
            #[link(name = "ws2_32")]
            unsafe extern "system" {
                fn setsockopt(
                    socket: usize,
                    level: i32,
                    name: i32,
                    value: *const u8,
                    length: i32,
                ) -> i32;
            }
            const SOL_SOCKET: i32 = 0xffff;
            const SO_RCVBUF: i32 = 0x1002;
            // SAFETY: a socket this player owns, and a value as long as said.
            unsafe {
                setsockopt(
                    socket.as_raw_socket() as usize,
                    SOL_SOCKET,
                    SO_RCVBUF,
                    (&raw const bytes).cast(),
                    4,
                );
            }
        }
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            unsafe extern "C" {
                fn setsockopt(
                    socket: i32,
                    level: i32,
                    name: i32,
                    value: *const std::ffi::c_void,
                    length: u32,
                ) -> i32;
            }
            #[cfg(target_os = "linux")]
            const OPTION: (i32, i32) = (1, 8);
            #[cfg(not(target_os = "linux"))]
            const OPTION: (i32, i32) = (0xffff, 0x1002);
            let (level, name) = OPTION;
            // SAFETY: a socket this player owns, and a value as long as said.
            unsafe {
                setsockopt(
                    socket.as_raw_fd(),
                    level,
                    name,
                    (&raw const bytes).cast(),
                    4,
                );
            }
        }
    }

    /// One RTP packet of `track`, arrived `now` into the run.
    fn rtp(packet: &[u8], track: &mut Track, now: Duration, totals: &Totals, seen: &mut Seen) {
        if packet.len() < 12 {
            return;
        }
        seen.packets += 1;
        seen.bytes += packet.len() as u64;
        totals
            .bytes
            .fetch_add(packet.len() as u64, Ordering::Relaxed);
        let sequence = u16::from_be_bytes([packet[2], packet[3]]);
        if let Some(expected) = track.expected {
            let gap = sequence.wrapping_sub(expected);
            if gap < 0x8000 {
                seen.lost += u64::from(gap);
            }
        }
        track.expected = Some(sequence.wrapping_add(1));
        let timestamp = u32::from_be_bytes([packet[4], packet[5], packet[6], packet[7]]);
        let unwrapped = match track.timestamp {
            None => 0,
            Some((last, at)) => at + i64::from(timestamp.wrapping_sub(last) as i32),
        };
        track.timestamp = Some((timestamp, unwrapped));
        track.newest = track.newest.max(unwrapped);
        let marker = packet[1] & 0x80 != 0;
        if track.video && marker {
            seen.pictures += 1;
            if now >= WARM_UP {
                let lag = now.as_secs_f64() - track.newest as f64 / track.clock;
                seen.lag = Some(match seen.lag {
                    None => (lag, lag),
                    Some((min, max)) => (min.min(lag), max.max(lag)),
                });
            }
        }
    }

    /// Sends one request and reads its response, keeping what came after it.
    fn request(
        connection: &mut TcpStream,
        pending: &mut Vec<u8>,
        cseq: &mut u32,
        method: &str,
        url: &str,
        headers: &[(&str, String)],
    ) -> Result<String, String> {
        *cseq += 1;
        let mut text =
            format!("{method} {url} RTSP/1.0\r\nCSeq: {cseq}\r\nUser-Agent: rtsp_load\r\n");
        for (name, value) in headers {
            text.push_str(&format!("{name}: {value}\r\n"));
        }
        text.push_str("\r\n");
        connection
            .write_all(text.as_bytes())
            .map_err(|e| format!("{method}: {e}"))?;
        let mut chunk = [0u8; 4096];
        loop {
            if let Some(end) = pending.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&pending[..end]).into_owned();
                let length: usize = header(&head, "Content-Length")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                if pending.len() >= end + 4 + length {
                    let body =
                        String::from_utf8_lossy(&pending[end + 4..end + 4 + length]).into_owned();
                    pending.drain(..end + 4 + length);
                    if !head.starts_with("RTSP/1.0 200") {
                        return Err(format!("{method}: {}", head.lines().next().unwrap_or("")));
                    }
                    return Ok(head + "\r\n\r\n" + &body);
                }
            }
            let read = connection
                .read(&mut chunk)
                .map_err(|e| format!("{method}: {e}"))?;
            if read == 0 {
                return Err(format!("{method}: the server closed the connection"));
            }
            pending.extend_from_slice(&chunk[..read]);
        }
    }

    fn header<'a>(response: &'a str, name: &str) -> Option<&'a str> {
        response.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.trim().eq_ignore_ascii_case(name).then(|| value.trim())
        })
    }

    /// A media section of a description: its control, its clock rate, and
    /// whether it is video.
    type Section = (String, f64, bool);

    /// The base URL a description's controls are relative to, and its media
    /// sections.
    fn sections(url: &str, response: &str) -> Result<(String, Vec<Section>), String> {
        let base = header(response, "Content-Base")
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{}/", url.trim_end_matches('/')));
        let body = response.split("\r\n\r\n").nth(1).unwrap_or("");
        let mut sections = Vec::new();
        for line in body.lines() {
            if let Some(media) = line.strip_prefix("m=") {
                sections.push((String::new(), 90_000.0, media.starts_with("video")));
            } else if let Some(last) = sections.last_mut() {
                if let Some(control) = line.strip_prefix("a=control:") {
                    last.0 = control.trim().to_owned();
                } else if let Some(map) = line.strip_prefix("a=rtpmap:")
                    && let Some(clock) = map.split('/').nth(1).and_then(|c| c.trim().parse().ok())
                {
                    last.1 = clock;
                }
            }
        }
        if sections.is_empty() {
            return Err("the description has no media".into());
        }
        Ok((base, sections))
    }
}
