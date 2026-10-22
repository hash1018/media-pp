//! RTSP Digest authentication (RFC 2617, as RTSP 1.0 uses it):
//! [`RtspDigest`], and the MD5 it is computed with.

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;

/// The realm every challenge names; a password stored as its Digest hash
/// is hashed with it.
pub(crate) const REALM: &str = "media-pp";

/// What an RTSP player presented answering a `Digest` challenge: who it
/// says it is, and a hash of its password that only that password makes.
///
/// It carries no password — the player never sends one — so it is checked
/// against the one the application keeps for the user, with
/// [`Self::verifies`]. Only an [`RtspServer`](crate::elements::RtspServer)
/// makes one, from a response to a challenge it gave on the same
/// connection; a response to any other is no credentials at all.
#[derive(Clone, PartialEq, Eq)]
pub struct RtspDigest {
    user: String,
    nonce: String,
    uri: String,
    response: String,
    method: String,
    /// `qop`, `nc` and `cnonce`, where the player sent them.
    qop: Option<(String, String, String)>,
}

impl RtspDigest {
    /// The user it says it is.
    pub fn user(&self) -> &str {
        &self.user
    }

    /// Whether `password` is the one it was made with: what decides that
    /// the player is [`Self::user`].
    pub fn verifies(&self, password: &str) -> bool {
        let ha1 = md5_hex(format!("{}:{REALM}:{password}", self.user).as_bytes());
        let ha2 = md5_hex(format!("{}:{}", self.method, self.uri).as_bytes());
        let expected = match &self.qop {
            Some((qop, nc, cnonce)) => {
                md5_hex(format!("{ha1}:{}:{nc}:{cnonce}:{qop}:{ha2}", self.nonce).as_bytes())
            }
            None => md5_hex(format!("{ha1}:{}:{ha2}", self.nonce).as_bytes()),
        };
        same(
            expected.as_bytes(),
            self.response.to_ascii_lowercase().as_bytes(),
        )
    }

    /// What `header` — an `Authorization` value — says as a response to the
    /// challenge with `nonce`, for a request of `method` on `path`; `None`
    /// where it is not a Digest response, is for another challenge or
    /// another realm, or names a URL of another path.
    pub(crate) fn parse(header: &str, method: &str, nonce: &str, path: &str) -> Option<Self> {
        let (scheme, rest) = header.trim().split_once(' ')?;
        if !scheme.eq_ignore_ascii_case("digest") {
            return None;
        }
        let mut user = None;
        let mut realm = None;
        let mut given_nonce = None;
        let mut uri = None;
        let mut response = None;
        let mut algorithm = None;
        let mut qop = None;
        let mut nc = None;
        let mut cnonce = None;
        for (name, value) in parameters(rest) {
            let slot = match name.to_ascii_lowercase().as_str() {
                "username" => &mut user,
                "realm" => &mut realm,
                "nonce" => &mut given_nonce,
                "uri" => &mut uri,
                "response" => &mut response,
                "algorithm" => &mut algorithm,
                "qop" => &mut qop,
                "nc" => &mut nc,
                "cnonce" => &mut cnonce,
                _ => continue,
            };
            *slot = Some(value);
        }
        if realm.as_deref() != Some(REALM)
            || given_nonce.as_deref() != Some(nonce)
            || algorithm.is_some_and(|a| !a.eq_ignore_ascii_case("md5"))
        {
            return None;
        }
        let uri = uri?;
        // The URL the hash was made over names the path asked for — a
        // player's track URLs, a trailing `/`, an absolute URL or a bare path
        // alike — so that a response cannot be carried to another path.
        if super::message::target(&uri).map(|target| target.path) != Some(path.to_owned())
            && uri.trim_matches('/') != path
        {
            return None;
        }
        let qop = match qop {
            Some(qop) if qop.eq_ignore_ascii_case("auth") => Some((qop, nc?, cnonce?)),
            Some(_) => return None,
            None => None,
        };
        Some(Self {
            user: user?,
            nonce: nonce.to_owned(),
            uri,
            response: response?,
            method: method.to_owned(),
            qop,
        })
    }
}

impl std::fmt::Debug for RtspDigest {
    /// Names the user, never the hash.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RtspDigest {{ user: {:?}, .. }}", self.user)
    }
}

/// A connection's own challenge value: never the same twice, and not to be
/// guessed from another.
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
fn parameters(text: &str) -> Vec<(String, String)> {
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

/// Whether two strings are the same, taking as long whichever byte differs.
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |diff, (x, y)| diff | (x ^ y)) == 0
}

pub(super) fn md5_hex(data: &[u8]) -> String {
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

    /// RFC 2617's own example, but in this server's realm and over RTSP:
    /// computed here as a player would, with and without `qop`.
    fn response(password: &str, qop: Option<(&str, &str)>) -> String {
        let ha1 = md5_hex(format!("Mufasa:{REALM}:{password}").as_bytes());
        let ha2 = md5_hex(b"DESCRIBE:rtsp://127.0.0.1:8554/cam");
        match qop {
            Some((nc, cnonce)) => {
                md5_hex(format!("{ha1}:abc123:{nc}:{cnonce}:auth:{ha2}").as_bytes())
            }
            None => md5_hex(format!("{ha1}:abc123:{ha2}").as_bytes()),
        }
    }

    #[test]
    fn a_response_verifies_with_its_password_alone() {
        let plain = format!(
            "Digest username=\"Mufasa\", realm=\"{REALM}\", nonce=\"abc123\", uri=\"rtsp://127.0.0.1:8554/cam\", response=\"{}\"",
            response("Circle Of Life", None)
        );
        let digest = RtspDigest::parse(&plain, "DESCRIBE", "abc123", "cam").unwrap();
        assert_eq!(digest.user(), "Mufasa");
        assert!(digest.verifies("Circle Of Life"));
        assert!(!digest.verifies("Circle of Life"));
        assert!(!format!("{digest:?}").contains(&response("Circle Of Life", None)));

        let with_qop = format!(
            "Digest username=\"Mufasa\",realm=\"{REALM}\",nonce=\"abc123\",uri=\"rtsp://127.0.0.1:8554/cam\",qop=auth,nc=00000001,cnonce=\"0a4f113b\",algorithm=MD5,response=\"{}\"",
            response("Circle Of Life", Some(("00000001", "0a4f113b")))
        );
        let digest = RtspDigest::parse(&with_qop, "DESCRIBE", "abc123", "cam").unwrap();
        assert!(digest.verifies("Circle Of Life"));
        // Made for DESCRIBE; offered for SETUP it is another hash.
        let digest = RtspDigest::parse(&with_qop, "SETUP", "abc123", "cam").unwrap();
        assert!(!digest.verifies("Circle Of Life"));
    }

    #[test]
    fn a_response_to_another_challenge_or_path_is_none() {
        let header = |nonce: &str, realm: &str, uri: &str| {
            format!(
                "Digest username=\"u\", realm=\"{realm}\", nonce=\"{nonce}\", uri=\"{uri}\", response=\"00\""
            )
        };
        let url = "rtsp://127.0.0.1:8554/cam";
        assert!(
            RtspDigest::parse(&header("abc123", REALM, url), "PLAY", "abc123", "cam").is_some()
        );
        assert!(
            RtspDigest::parse(
                &header("abc123", REALM, &format!("{url}/streamid=1")),
                "SETUP",
                "abc123",
                "cam"
            )
            .is_some(),
            "a track's URL is its path's"
        );
        assert!(RtspDigest::parse(&header("other", REALM, url), "PLAY", "abc123", "cam").is_none());
        assert!(
            RtspDigest::parse(&header("abc123", "elsewhere", url), "PLAY", "abc123", "cam")
                .is_none()
        );
        assert!(
            RtspDigest::parse(&header("abc123", REALM, url), "PLAY", "abc123", "live").is_none()
        );
        assert!(RtspDigest::parse("Basic dTpw", "PLAY", "abc123", "cam").is_none());
    }

    #[test]
    fn nonces_differ() {
        assert_ne!(nonce(), nonce());
        assert_eq!(nonce().len(), 32);
    }
}
