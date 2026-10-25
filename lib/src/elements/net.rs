//! Reading a socket with a deadline: [`recv_within`] for UDP,
//! [`read_within`] for TCP.
//!
//! A read with a read timeout loses data on Windows. What arrives as the
//! timeout runs out is gone: a datagram never handed to that receive nor to
//! the next, a TCP stream's bytes taken from it and never handed over — one
//! write's worth missing from what is otherwise whole and in order. Read
//! every few milliseconds with little arriving between — a player's socket
//! between the pictures of a stream that sends little — a stream of thirty
//! pictures a second lost a picture's packet every few seconds on an idle
//! machine, which only a resend made good, and under load not always.
//! There, the wait is `WSAPoll`'s, and the read that follows has data
//! waiting: no read ever times out. Elsewhere a read timeout loses nothing,
//! and the read waits itself.

use std::io::{self, Read};
use std::net::{SocketAddr, TcpStream, UdpSocket};
use std::time::Duration;

/// Waits up to `wait` for a datagram on `socket` and reads it into
/// `buffer`. An error of kind [`TimedOut`](io::ErrorKind::TimedOut) where
/// none came; any other is the receive's own — a Windows socket's
/// `ConnectionReset` after an earlier send went nowhere among them.
///
/// The socket is to have no read timeout of its own, and one thread
/// reading it: what the wait saw waiting is what the receive takes.
pub(crate) fn recv_within(
    socket: &UdpSocket,
    buffer: &mut [u8],
    wait: Duration,
) -> io::Result<(usize, SocketAddr)> {
    #[cfg(windows)]
    if !readable_within(socket, Some(wait))? {
        return Err(io::ErrorKind::TimedOut.into());
    }
    #[cfg(not(windows))]
    socket.set_read_timeout(Some(wait.max(Duration::from_micros(1))))?;
    socket.recv_from(buffer).map_err(timed_out)
}

/// Waits up to `wait` — without end where `None` — for what `stream` has
/// to read, and reads it into `buffer`: `Ok(0)` once the peer has closed
/// it, an error of kind [`TimedOut`](io::ErrorKind::TimedOut) where nothing
/// came. As [`recv_within`], with no read timeout of the stream's own and
/// one thread reading it.
pub(crate) fn read_within(
    stream: &TcpStream,
    buffer: &mut [u8],
    wait: Option<Duration>,
) -> io::Result<usize> {
    #[cfg(windows)]
    if !readable_within(stream, wait)? {
        return Err(io::ErrorKind::TimedOut.into());
    }
    #[cfg(not(windows))]
    stream.set_read_timeout(wait.map(|wait| wait.max(Duration::from_micros(1))))?;
    let mut stream = stream;
    stream.read(buffer).map_err(timed_out)
}

/// A read with nothing to read, as [`TimedOut`](io::ErrorKind::TimedOut)
/// whichever way the system says it.
fn timed_out(error: io::Error) -> io::Error {
    match error.kind() {
        io::ErrorKind::WouldBlock => io::ErrorKind::TimedOut.into(),
        _ => error,
    }
}

/// Whether data, an end or an error waits on `socket` within `wait`, or
/// ever where `None`.
#[cfg(windows)]
fn readable_within(
    socket: &impl std::os::windows::io::AsRawSocket,
    wait: Option<Duration>,
) -> io::Result<bool> {
    /// `WSAPOLLFD`.
    #[repr(C)]
    struct PollFd {
        fd: usize,
        events: i16,
        revents: i16,
    }
    const POLLRDNORM: i16 = 0x0100;
    #[link(name = "ws2_32")]
    unsafe extern "system" {
        fn WSAPoll(fds: *mut PollFd, count: u32, timeout: i32) -> i32;
    }

    let mut fd = PollFd {
        fd: socket.as_raw_socket() as usize,
        events: POLLRDNORM,
        revents: 0,
    };
    // In whole milliseconds, as a read timeout was: a wait under one is
    // one, not a look that returns at once and leaves its caller spinning.
    let timeout = wait.map_or(-1, |wait| {
        i32::try_from(wait.as_micros().div_ceil(1000)).unwrap_or(i32::MAX)
    });
    // SAFETY: one `WSAPOLLFD`, for a socket `socket` keeps open throughout.
    match unsafe { WSAPoll(&mut fd, 1, timeout) } {
        // `POLLHUP` or `POLLERR` as well: the read says what it was.
        ready if ready > 0 => Ok(true),
        0 => Ok(false),
        _ => Err(io::Error::last_os_error()),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::net::TcpListener;
    use std::time::Instant;

    use super::*;

    /// A writer sending a record at a time, and its reader: whole records
    /// read, or `None` where its wait ran out.
    type Open = fn() -> (
        Box<dyn FnMut(u32) + Send>,
        Box<dyn FnMut() -> Option<Vec<u32>>>,
    );

    /// Runs `pairs` writers and readers side by side, each writer sending
    /// `count` numbered records `gap` apart, then one more to end on: what
    /// each reader read, in order, and how many times its waits ran out.
    fn side_by_side(pairs: usize, count: u32, gap: Duration, open: Open) -> Vec<(Vec<u32>, u32)> {
        let runs: Vec<_> = (0..pairs)
            .map(|_| {
                std::thread::spawn(move || {
                    let (mut send, mut read) = open();
                    let writer = std::thread::spawn(move || {
                        let mut next = Instant::now();
                        for n in 0..=count {
                            send(n);
                            next += gap;
                            if let Some(left) = next.checked_duration_since(Instant::now()) {
                                std::thread::sleep(left);
                            }
                        }
                    });
                    let mut seen = Vec::new();
                    let mut ran_out = 0;
                    let until = Instant::now() + Duration::from_secs(30);
                    'reading: while Instant::now() < until {
                        match read() {
                            Some(records) => {
                                for record in records {
                                    if record == count {
                                        break 'reading;
                                    }
                                    seen.push(record);
                                }
                            }
                            None => ran_out += 1,
                        }
                    }
                    writer.join().unwrap();
                    (seen, ran_out)
                })
            })
            .collect();
        runs.into_iter().map(|run| run.join().unwrap()).collect()
    }

    const WAIT: Duration = Duration::from_millis(1);

    /// A datagram every millisecond read with waits of a millisecond, so
    /// that waits keep running out as datagrams come: every one is read.
    /// With a read timeout instead, Windows lost one or more in nearly every
    /// run.
    #[test]
    fn datagrams_coming_as_waits_run_out_are_all_read() {
        const SENT: u32 = 3000;
        let runs = side_by_side(3, SENT, Duration::from_millis(1), || {
            let receiver = UdpSocket::bind("127.0.0.1:0").unwrap();
            let to = receiver.local_addr().unwrap();
            let sender = UdpSocket::bind("127.0.0.1:0").unwrap();
            let mut buffer = [0; 16];
            (
                Box::new(move |n: u32| {
                    sender.send_to(&n.to_be_bytes(), to).unwrap();
                }),
                Box::new(move || match recv_within(&receiver, &mut buffer, WAIT) {
                    Ok((4, _)) => Some(vec![u32::from_be_bytes(buffer[..4].try_into().unwrap())]),
                    Ok(_) => Some(Vec::new()),
                    Err(error) => {
                        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
                        None
                    }
                }),
            )
        });
        for (seen, ran_out) in runs {
            assert_eq!(
                seen,
                (0..SENT).collect::<Vec<_>>(),
                "every datagram read; waits ran out {ran_out} times"
            );
        }
    }

    /// The same of a TCP stream, its records written apart: every byte is
    /// read, in order. A read timeout lost a write's worth of one only once
    /// in some ten thousand waits run out, too seldom for a test this short
    /// to show; the datagrams' shows what the wait keeps, and this holds the
    /// stream's reads to it.
    #[test]
    fn a_stream_coming_as_waits_run_out_is_read_whole() {
        const SENT: u32 = 2000;
        let runs = side_by_side(3, SENT, Duration::from_micros(1500), || {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let mut writer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            writer.set_nodelay(true).unwrap();
            let (reader, _) = listener.accept().unwrap();
            let mut pending = Vec::new();
            let mut buffer = [0; 4096];
            (
                Box::new(move |n: u32| writer.write_all(&n.to_be_bytes()).unwrap()),
                Box::new(
                    move || match read_within(&reader, &mut buffer, Some(WAIT)) {
                        Ok(read) => {
                            assert_ne!(read, 0, "the writer closed early");
                            pending.extend_from_slice(&buffer[..read]);
                            let whole = pending.len() / 4 * 4;
                            let records = pending[..whole]
                                .chunks(4)
                                .map(|record| u32::from_be_bytes(record.try_into().unwrap()))
                                .collect();
                            pending.drain(..whole);
                            Some(records)
                        }
                        Err(error) => {
                            assert_eq!(error.kind(), io::ErrorKind::TimedOut);
                            None
                        }
                    },
                ),
            )
        });
        for (seen, ran_out) in runs {
            assert_eq!(
                seen,
                (0..SENT).collect::<Vec<_>>(),
                "every byte read, in order; waits ran out {ran_out} times"
            );
        }
    }

    #[test]
    fn nothing_coming_it_times_out_within_about_the_wait() {
        let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let stream = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let wait = Duration::from_millis(20);
        let reads: [&mut dyn FnMut() -> io::Result<()>; 2] = [
            &mut || recv_within(&socket, &mut [0; 16], wait).map(|_| ()),
            &mut || read_within(&stream, &mut [0; 16], Some(wait)).map(|_| ()),
        ];
        for read in reads {
            let began = Instant::now();
            let error = read().unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::TimedOut);
            let took = began.elapsed();
            assert!(took >= Duration::from_millis(15), "{took:?}");
            assert!(took < Duration::from_secs(1), "{took:?}");
        }
    }
}
