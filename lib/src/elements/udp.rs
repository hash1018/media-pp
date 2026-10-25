//! Reading a UDP socket with a deadline: [`recv_within`].
//!
//! A receive with a read timeout loses datagrams on Windows. One that
//! arrives as the timeout runs out is gone: never handed to that receive,
//! nor to the next. Read every few milliseconds with nothing else arriving
//! between — a player's socket between the pictures of a stream that sends
//! little — a stream of thirty pictures a second lost a picture's packet
//! every few seconds on an idle machine, which only a resend made good, and
//! under load not always. There, the wait is `WSAPoll`'s, and the receive
//! that follows has a datagram waiting: no receive ever times out.

use std::io;
use std::net::{SocketAddr, UdpSocket};
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
    if !readable_within(socket, wait)? {
        return Err(io::ErrorKind::TimedOut.into());
    }
    // A read timeout loses nothing elsewhere: the receive waits itself.
    #[cfg(not(windows))]
    socket.set_read_timeout(Some(wait.max(Duration::from_micros(1))))?;
    socket
        .recv_from(buffer)
        .map_err(|error| match error.kind() {
            io::ErrorKind::WouldBlock => io::ErrorKind::TimedOut.into(),
            _ => error,
        })
}

/// Whether a datagram, or an error, waits on `socket` within `wait`.
#[cfg(windows)]
fn readable_within(socket: &UdpSocket, wait: Duration) -> io::Result<bool> {
    use std::os::windows::io::AsRawSocket;

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
    let timeout = i32::try_from(wait.as_micros().div_ceil(1000)).unwrap_or(i32::MAX);
    // SAFETY: one `WSAPOLLFD`, for a socket `socket` keeps open throughout.
    match unsafe { WSAPoll(&mut fd, 1, timeout) } {
        // `POLLERR` or `POLLHUP` as well: the receive says what went wrong.
        ready if ready > 0 => Ok(true),
        0 => Ok(false),
        _ => Err(io::Error::last_os_error()),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    /// A datagram every millisecond read with waits of a millisecond, so
    /// that waits keep running out as datagrams come: every one is read.
    /// With a read timeout instead, Windows lost one or more in nearly every
    /// run of five seconds.
    #[test]
    fn datagrams_coming_as_waits_run_out_are_all_read() {
        const SENT: u32 = 3000;
        let pairs: Vec<_> = (0..3)
            .map(|_| {
                std::thread::spawn(|| {
                    let receiver = UdpSocket::bind("127.0.0.1:0").unwrap();
                    let to = receiver.local_addr().unwrap();
                    let sender = std::thread::spawn(move || {
                        let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
                        let mut next = Instant::now();
                        for n in 0..=SENT {
                            socket.send_to(&n.to_be_bytes(), to).unwrap();
                            next += Duration::from_millis(1);
                            if let Some(left) = next.checked_duration_since(Instant::now()) {
                                std::thread::sleep(left);
                            }
                        }
                    });
                    let mut buffer = [0; 16];
                    let mut read = 0;
                    let mut ran_out = 0;
                    let until = Instant::now() + Duration::from_secs(30);
                    while Instant::now() < until {
                        match recv_within(&receiver, &mut buffer, Duration::from_millis(1)) {
                            Ok((4, _)) => {
                                if u32::from_be_bytes(buffer[..4].try_into().unwrap()) == SENT {
                                    break;
                                }
                                read += 1;
                            }
                            Ok(_) => {}
                            Err(error) => {
                                assert_eq!(error.kind(), io::ErrorKind::TimedOut);
                                ran_out += 1;
                            }
                        }
                    }
                    sender.join().unwrap();
                    (read, ran_out)
                })
            })
            .collect();
        for pair in pairs {
            let (read, ran_out) = pair.join().unwrap();
            assert_eq!(
                read, SENT,
                "every datagram read; waits ran out {ran_out} times"
            );
        }
    }

    #[test]
    fn nothing_coming_it_times_out_within_about_the_wait() {
        let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        let began = Instant::now();
        let error = recv_within(&socket, &mut [0; 16], Duration::from_millis(20)).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        let took = began.elapsed();
        assert!(took >= Duration::from_millis(15), "{took:?}");
        assert!(took < Duration::from_secs(1), "{took:?}");
    }
}
