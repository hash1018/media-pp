//! RTSP authentication as both ends compute it: Digest (RFC 2617, as RTSP
//! 1.0 uses it) and Basic. The server checks what a player presents
//! (`RtspDigest`, in the server); a client answers the challenge a server
//! gives ([`Authenticator`]).

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;

/// A value never the same twice, and not to be guessed from another: a
/// server's challenge, a client's `cnonce`.
pub(crate) fn nonce() -> String {
    let state = RandomState::new();
    format!(
        "{:016x}{:016x}",
        state.hash_one(1u8),
        state.hash_one(std::time::Instant::now())
    )
}

/// `name=value` and `name="value"` pairs, comma-separated; a quoted value
/// may hold a comma.
pub(crate) fn parameters(text: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let mut rest = text;
    loop {
        rest = rest.trim_start_matches([' ', ',', '\t']);
        let Some((name, after)) = rest.split_once('=') else {
            return pairs;
        };
        let after = after.trim_start();
        let (value, next) = if let Some(quoted) = after.strip_prefix('"') {
            match quoted.find('"') {
                Some(end) => (&quoted[..end], &quoted[end + 1..]),
                None => return pairs,
            }
        } else {
            let end = after.find(',').unwrap_or(after.len());
            (after[..end].trim(), &after[end..])
        };
        pairs.push((name.trim().to_owned(), value.to_owned()));
        rest = next;
    }
}

/// What a client answers a server's `401` with, from then on: the scheme
/// the server asked for, with the user and password the client was given.
///
/// A Digest answer is a hash over the method and the URL of each request,
/// so it is made anew for each; one made with `qop=auth` counts its
/// requests, as the server checks.
pub(crate) struct Authenticator {
    user: String,
    password: String,
    scheme: Scheme,
}

enum Scheme {
    Basic,
    Digest {
        realm: String,
        nonce: String,
        opaque: Option<String>,
        /// `qop=auth` offered: the answer then counts its requests.
        qop: bool,
        count: u32,
    },
}

impl Authenticator {
    /// The answer to the challenges a `401` gave — its
    /// `WWW-Authenticate` headers — Digest where one offers it, since it
    /// sends no password; `None` where none is a scheme this answers.
    pub(crate) fn new<'a>(
        user: &str,
        password: &str,
        challenges: impl IntoIterator<Item = &'a str>,
    ) -> Option<Self> {
        let mut basic = false;
        let mut digest = None;
        for challenge in challenges {
            let (scheme, rest) = challenge
                .trim()
                .split_once(' ')
                .unwrap_or((challenge.trim(), ""));
            if scheme.eq_ignore_ascii_case("basic") {
                basic = true;
            } else if scheme.eq_ignore_ascii_case("digest") && digest.is_none() {
                digest = digest_challenge(rest);
            }
        }
        let scheme = digest.or(basic.then_some(Scheme::Basic))?;
        Some(Self {
            user: user.to_owned(),
            password: password.to_owned(),
            scheme,
        })
    }

    /// The `Authorization` header for a request of `method` on `uri`.
    pub(crate) fn authorization(&mut self, method: &str, uri: &str) -> String {
        match &mut self.scheme {
            Scheme::Basic => format!(
                "Basic {}",
                base64(format!("{}:{}", self.user, self.password).as_bytes())
            ),
            Scheme::Digest {
                realm,
                nonce,
                opaque,
                qop,
                count,
            } => {
                let ha1 = md5_hex(format!("{}:{realm}:{}", self.user, self.password).as_bytes());
                let ha2 = md5_hex(format!("{method}:{uri}").as_bytes());
                let mut header = format!(
                    "Digest username=\"{}\", realm=\"{realm}\", nonce=\"{nonce}\", uri=\"{uri}\"",
                    self.user
                );
                if *qop {
                    *count += 1;
                    let nc = format!("{count:08x}");
                    let cnonce = &nonce_prefix();
                    let response =
                        md5_hex(format!("{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}").as_bytes());
                    header.push_str(&format!(
                        ", response=\"{response}\", qop=auth, nc={nc}, cnonce=\"{cnonce}\""
                    ));
                } else {
                    let response = md5_hex(format!("{ha1}:{nonce}:{ha2}").as_bytes());
                    header.push_str(&format!(", response=\"{response}\""));
                }
                if let Some(opaque) = opaque {
                    header.push_str(&format!(", opaque=\"{opaque}\""));
                }
                header.push_str(", algorithm=MD5");
                header
            }
        }
    }
}

/// A Digest challenge this can answer: MD5, and `qop=auth` or none.
fn digest_challenge(text: &str) -> Option<Scheme> {
    let mut realm = None;
    let mut nonce = None;
    let mut opaque = None;
    let mut qop = None;
    let mut algorithm = None;
    for (name, value) in parameters(text) {
        match name.to_ascii_lowercase().as_str() {
            "realm" => realm = Some(value),
            "nonce" => nonce = Some(value),
            "opaque" => opaque = Some(value),
            "qop" => qop = Some(value),
            "algorithm" => algorithm = Some(value),
            _ => {}
        }
    }
    if algorithm.is_some_and(|a| !a.eq_ignore_ascii_case("md5")) {
        return None;
    }
    let qop = match qop {
        None => false,
        Some(offered) => {
            if !offered
                .split(',')
                .any(|qop| qop.trim().eq_ignore_ascii_case("auth"))
            {
                return None;
            }
            true
        }
    };
    Some(Scheme::Digest {
        realm: realm?,
        nonce: nonce?,
        opaque,
        qop,
        count: 0,
    })
}

/// A client nonce: sixteen hex digits are as many as any server checks.
fn nonce_prefix() -> String {
    nonce()[..16].to_owned()
}

/// Whether two strings are the same, taking as long whichever byte differs.
#[cfg(feature = "rtsp-server")]
pub(crate) fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |diff, (x, y)| diff | (x ^ y)) == 0
}

/// Standard base64 with padding — Basic's encoding of `user:password`.
fn base64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let bytes = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let group = (u32::from(bytes[0]) << 16) | (u32::from(bytes[1]) << 8) | u32::from(bytes[2]);
        for index in 0..4 {
            if index <= chunk.len() {
                out.push(char::from(
                    ALPHABET[((group >> (18 - 6 * index)) & 0x3f) as usize],
                ));
            } else {
                out.push('=');
            }
        }
    }
    out
}

pub(crate) fn md5_hex(data: &[u8]) -> String {
    md5(data).iter().map(|byte| format!("{byte:02x}")).collect()
}

/// MD5 (RFC 1321) — what RTSP Digest is computed with, and nothing else
/// here: it is not used to keep anything secret on its own.
fn md5(data: &[u8]) -> [u8; 16] {
    const SHIFTS: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let constants: [u32; 64] =
        std::array::from_fn(|i| ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32);
    let mut state: [u32; 4] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64).wrapping_mul(8)).to_le_bytes());
    for block in message.as_chunks::<64>().0 {
        let words: [u32; 16] =
            std::array::from_fn(|i| u32::from_le_bytes(block.as_chunks::<4>().0[i]));
        let [mut a, mut b, mut c, mut d] = state;
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let rotated = a
                .wrapping_add(f)
                .wrapping_add(constants[i])
                .wrapping_add(words[g])
                .rotate_left(SHIFTS[i]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(rotated);
        }
        for (word, add) in state.iter_mut().zip([a, b, c, d]) {
            *word = word.wrapping_add(add);
        }
    }
    let mut out = [0u8; 16];
    for (chunk, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        *chunk = word.to_le_bytes();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn md5_is_rfc_1321s() {
        for (input, digest) in [
            ("", "d41d8cd98f00b204e9800998ecf8427e"),
            ("abc", "900150983cd24fb0d6963f7d28e17f72"),
            ("message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
            (
                "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
                "57edf4a22be3c955ac49da2e2107b67a",
            ),
        ] {
            assert_eq!(md5_hex(input.as_bytes()), digest, "{input:?}");
        }
    }

    #[test]
    fn nonces_differ() {
        assert_ne!(nonce(), nonce());
        assert_eq!(nonce().len(), 32);
    }

    #[test]
    fn basic_is_the_user_and_password_in_base64() {
        // RFC 7617's example.
        let mut auth = Authenticator::new("Aladdin", "open sesame", ["Basic realm=\"x\""]).unwrap();
        assert_eq!(
            auth.authorization("DESCRIBE", "rtsp://h/cam"),
            "Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ=="
        );
        assert_eq!(base64(b"Ma"), "TWE=");
        assert_eq!(base64(b"M"), "TQ==");
    }

    /// RFC 2617's own example, answered as a client: the response it gives.
    #[test]
    fn digest_answers_as_rfc_2617s_example() {
        let challenge = "Digest realm=\"testrealm@host.com\", qop=\"auth,auth-int\", \
                         nonce=\"dcd98b7102dd2f0e8b11d0f600bfb0c093\", \
                         opaque=\"5ccc069c403ebaf9f0171e9517f40e41\"";
        let mut auth =
            Authenticator::new("Mufasa", "Circle Of Life", ["Basic realm=\"x\"", challenge])
                .expect("Digest is answered");
        let header = auth.authorization("GET", "/dir/index.html");
        let Scheme::Digest { count, .. } = &auth.scheme else {
            panic!("Digest is preferred over Basic");
        };
        assert_eq!(*count, 1);
        // The cnonce is the client's own; recompute the response with it.
        let fields: Vec<(String, String)> = parameters(header.strip_prefix("Digest ").unwrap());
        let field = |name: &str| {
            fields
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
                .unwrap()
        };
        let ha1 = md5_hex(b"Mufasa:testrealm@host.com:Circle Of Life");
        let ha2 = md5_hex(b"GET:/dir/index.html");
        let expected = md5_hex(
            format!(
                "{ha1}:dcd98b7102dd2f0e8b11d0f600bfb0c093:00000001:{}:auth:{ha2}",
                field("cnonce")
            )
            .as_bytes(),
        );
        assert_eq!(field("response"), expected);
        assert_eq!(field("nc"), "00000001");
        assert_eq!(field("opaque"), "5ccc069c403ebaf9f0171e9517f40e41");
        assert!(auth.authorization("GET", "/").contains("nc=00000002"));
    }

    #[test]
    fn a_challenge_this_cannot_answer_is_none() {
        assert!(
            Authenticator::new(
                "u",
                "p",
                ["Digest realm=\"r\", nonce=\"n\", algorithm=SHA-256"]
            )
            .is_none()
        );
        assert!(
            Authenticator::new(
                "u",
                "p",
                ["Digest realm=\"r\", nonce=\"n\", qop=\"auth-int\""]
            )
            .is_none()
        );
        assert!(Authenticator::new("u", "p", ["Bearer realm=\"r\""]).is_none());
        assert!(Authenticator::new("u", "p", Vec::<&str>::new()).is_none());
    }
}
