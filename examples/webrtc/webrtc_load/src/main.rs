//! A load test for a WHEP server: as many WebRTC players as asked, each a
//! thread with a `str0m` connection of its own that plays the stream and
//! counts what it is sent — pictures, bytes, the gaps where packets were
//! lost, and how steadily its pictures arrive — and a summary at the end.
//! `docs/benchmarks/streaming` is made with it, against the
//! `webrtc_server` example.
//!
//! Each player posts an offer for a picture and a sound to the WHEP URL, as
//! a browser's player does, and decodes nothing: what it costs is the
//! connection's own — ICE, DTLS, SRTP — and taking RTP apart, which a real
//! player pays too. It ends its session with a `DELETE` when the run ends.
//!
//! How steadily pictures arrive is a picture's arrival less the newest
//! timestamp handed over yet, its spread — the largest less the smallest —
//! over the run after a player's first three seconds.
//!
//!     cargo run --release -p webrtc_load -- http://127.0.0.1:8080/file/whep 100 [--seconds 30]

fn main() -> Result<(), media_pp::Report> {
    Ok(example::run()?)
}

mod example {
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpStream, ToSocketAddrs, UdpSocket};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    use str0m::change::SdpAnswer;
    use str0m::media::{Direction, MediaKind};
    use str0m::net::{Protocol, Receive};
    use str0m::{Candidate, Event, Input, Output, Rtc};

    const USAGE: &str = "usage: webrtc_load <http://host:port/path/whep> <players> [--seconds N]";

    /// How long after it connects a player's pictures count towards its
    /// spread: past the catching up it joins with.
    const WARM_UP: Duration = Duration::from_secs(3);

    struct Args {
        url: String,
        players: usize,
        seconds: u64,
    }

    fn args() -> Option<Args> {
        let mut positional = Vec::new();
        let mut seconds = 30;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--seconds" => seconds = args.next()?.parse().ok()?,
                _ => positional.push(arg),
            }
        }
        let [url, players] = <[String; 2]>::try_from(positional).ok()?;
        Some(Args {
            url,
            players: players.parse().ok().filter(|&n| n > 0)?,
            seconds,
        })
    }

    #[derive(Default)]
    struct Totals {
        bytes: AtomicU64,
        playing: AtomicU64,
    }

    /// What one player saw.
    #[derive(Default)]
    struct Seen {
        pictures: u64,
        sounds: u64,
        bytes: u64,
        /// Pictures that arrived after a gap: some of what came before them
        /// was lost.
        gaps: u64,
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
            "{} players of {} for {} s",
            args.players, args.url, args.seconds
        );
        let began = Instant::now();
        let players: Vec<_> = (0..args.players)
            .map(|n| {
                let (url, stop, totals) =
                    (args.url.clone(), Arc::clone(&stop), Arc::clone(&totals));
                thread::sleep(Duration::from_millis(5));
                thread::Builder::new()
                    .name(format!("player-{n}"))
                    .spawn(move || play(&url, &stop, &totals))
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
            "received: {:.1} Mbit/s in all, {:.2} to {:.2} per player",
            mbit.iter().sum::<f64>(),
            mbit.iter().copied().fold(f64::INFINITY, f64::min),
            mbit.iter().copied().fold(0.0, f64::max)
        );
        println!(
            "pictures: {} in all, {} after a gap; {} players had a gap; sounds: {}",
            played.iter().map(|s| s.pictures).sum::<u64>(),
            played.iter().map(|s| s.gaps).sum::<u64>(),
            played.iter().filter(|s| s.gaps > 0).count(),
            played.iter().map(|s| s.sounds).sum::<u64>()
        );
        println!(
            "lag spread: median {:.1} ms, p99 {:.1} ms, max {:.1} ms",
            at(0.5),
            at(0.99),
            at(1.0)
        );
    }

    /// Asks for a receive buffer of `bytes`. Without a crate for it, the
    /// call itself.
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

    /// Waits up to `wait` for a datagram and reads it. Not with a read
    /// timeout on Windows: a datagram that comes as one runs out is lost,
    /// which a stream sending little — a picture every 33 ms — met every
    /// few seconds, and counted as the server's gap. There the wait is
    /// `WSAPoll`'s, the call itself, and the receive after it has a
    /// datagram waiting.
    fn recv_within(
        socket: &UdpSocket,
        buffer: &mut [u8],
        wait: Duration,
    ) -> std::io::Result<(usize, SocketAddr)> {
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawSocket;
            #[repr(C)]
            struct PollFd {
                fd: usize,
                events: i16,
                revents: i16,
            }
            #[link(name = "ws2_32")]
            unsafe extern "system" {
                fn WSAPoll(fds: *mut PollFd, count: u32, timeout: i32) -> i32;
            }
            const POLLRDNORM: i16 = 0x0100;
            let mut fd = PollFd {
                fd: socket.as_raw_socket() as usize,
                events: POLLRDNORM,
                revents: 0,
            };
            let timeout = i32::try_from(wait.as_micros().div_ceil(1000)).unwrap_or(i32::MAX);
            // SAFETY: one `WSAPOLLFD`, for a socket this player owns.
            if unsafe { WSAPoll(&mut fd, 1, timeout) } == 0 {
                return Err(std::io::ErrorKind::TimedOut.into());
            }
        }
        #[cfg(not(windows))]
        socket.set_read_timeout(Some(wait))?;
        socket.recv_from(buffer)
    }

    fn play(url: &str, stop: &AtomicBool, totals: &Totals) -> Seen {
        let mut seen = Seen::default();
        if let Err(error) = session(url, stop, totals, &mut seen) {
            seen.failed = Some(error);
        }
        seen
    }

    /// The HTTP server's address and the request path of `url`.
    fn split(url: &str) -> Result<(SocketAddr, String), String> {
        let rest = url.strip_prefix("http://").ok_or("not an http:// URL")?;
        let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
        let addr = authority
            .to_socket_addrs()
            .map_err(|e| e.to_string())?
            .next()
            .ok_or("no address")?;
        Ok((addr, format!("/{path}")))
    }

    /// One HTTP request, and its response's head and body: the response is
    /// read to the connection's end, so each request asks for that with
    /// `Connection: close` — a server that keeps connections open, as most
    /// do, would otherwise leave this waiting.
    fn http(addr: SocketAddr, request: &str) -> Result<(String, String), String> {
        let mut stream = TcpStream::connect(addr).map_err(|e| format!("connect: {e}"))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .map_err(|e| e.to_string())?;
        stream
            .write_all(request.as_bytes())
            .map_err(|e| e.to_string())?;
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .map_err(|e| e.to_string())?;
        let (head, body) = response.split_once("\r\n\r\n").unwrap_or((&response, ""));
        Ok((head.to_owned(), body.to_owned()))
    }

    fn session(
        url: &str,
        stop: &AtomicBool,
        totals: &Totals,
        seen: &mut Seen,
    ) -> Result<(), String> {
        let (server, path) = split(url)?;
        // The interface the server is reached on, for the player's candidate.
        let local_ip = TcpStream::connect(server)
            .and_then(|probe| probe.local_addr())
            .map_err(|e| format!("connect: {e}"))?
            .ip();
        let socket = UdpSocket::bind(SocketAddr::new(local_ip, 0)).map_err(|e| e.to_string())?;
        // As a browser does: a 1080p keyframe arrives at once, and is several
        // times a socket's default buffer.
        receive_buffer(&socket, 8 * 1024 * 1024);
        let local = socket.local_addr().map_err(|e| e.to_string())?;
        let mut rtc = Rtc::builder().build(Instant::now());
        rtc.add_local_candidate(Candidate::host(local, "udp").map_err(|e| e.to_string())?);
        let mut api = rtc.sdp_api();
        let video = api.add_media(MediaKind::Video, Direction::RecvOnly, None, None, None);
        api.add_media(MediaKind::Audio, Direction::RecvOnly, None, None, None);
        let (offer, pending) = api.apply().ok_or("no offer")?;
        let offer = offer.to_sdp_string();
        let (head, body) = http(
            server,
            &format!(
                "POST {path} HTTP/1.1\r\nHost: {server}\r\nConnection: close\r\nContent-Type: application/sdp\r\nContent-Length: {}\r\n\r\n{offer}",
                offer.len()
            ),
        )?;
        if !head.starts_with("HTTP/1.1 201") {
            return Err(format!("WHEP: {}", head.lines().next().unwrap_or("")));
        }
        let location = head.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("location")
                .then(|| value.trim().to_owned())
        });
        let answer = SdpAnswer::from_sdp_string(&body).map_err(|e| e.to_string())?;
        rtc.sdp_api()
            .accept_answer(pending, answer)
            .map_err(|e| e.to_string())?;

        let mut buffer = vec![0u8; 65_536];
        let mut connected: Option<Instant> = None;
        let mut newest = 0.0f64;
        let result = loop {
            if stop.load(Ordering::Relaxed) {
                break Ok(());
            }
            let mut wait = Duration::from_millis(50);
            loop {
                match rtc.poll_output() {
                    Ok(Output::Timeout(at)) => {
                        wait = wait.min(at.saturating_duration_since(Instant::now()));
                        break;
                    }
                    Ok(Output::Transmit(transmit)) => {
                        let _ = socket.send_to(&transmit.contents, transmit.destination);
                    }
                    Ok(Output::Event(Event::Connected)) => {
                        connected = Some(Instant::now());
                        totals.playing.fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(Output::Event(Event::MediaData(data))) => {
                        seen.bytes += data.data.len() as u64;
                        totals
                            .bytes
                            .fetch_add(data.data.len() as u64, Ordering::Relaxed);
                        if data.mid != video {
                            seen.sounds += 1;
                            continue;
                        }
                        seen.pictures += 1;
                        if !data.contiguous {
                            seen.gaps += 1;
                        }
                        let at = data.time.numer() as f64 / data.time.denom() as f64;
                        newest = newest.max(at);
                        if let Some(since) = connected
                            && since.elapsed() >= WARM_UP
                        {
                            let lag = since.elapsed().as_secs_f64() - newest;
                            seen.lag = Some(match seen.lag {
                                None => (lag, lag),
                                Some((min, max)) => (min.min(lag), max.max(lag)),
                            });
                        }
                    }
                    Ok(Output::Event(Event::IceConnectionStateChange(
                        str0m::IceConnectionState::Disconnected,
                    ))) => {
                        break;
                    }
                    Ok(Output::Event(_)) => {}
                    Err(error) => return Err(error.to_string()),
                }
            }
            if !rtc.is_alive() {
                break Err("the connection closed".into());
            }
            match recv_within(&socket, &mut buffer, wait.max(Duration::from_millis(1))) {
                Ok((length, source)) => {
                    if let Ok(contents) = buffer[..length].try_into() {
                        let _ = rtc.handle_input(Input::Receive(
                            Instant::now(),
                            Receive {
                                proto: Protocol::Udp,
                                source,
                                destination: local,
                                contents,
                            },
                        ));
                    }
                }
                Err(_) => {
                    let _ = rtc.handle_input(Input::Timeout(Instant::now()));
                }
            }
        };
        if connected.is_some() {
            totals.playing.fetch_sub(1, Ordering::Relaxed);
        } else if result.is_ok() {
            return Err("never connected".into());
        }
        if let Some(location) = location {
            let _ = http(
                server,
                &format!(
                    "DELETE {location} HTTP/1.1\r\nHost: {server}\r\nConnection: close\r\n\r\n"
                ),
            );
        }
        result
    }
}
