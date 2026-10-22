//! Who may play or publish a server's paths: [`Authorize`], shared by the
//! RTSP, WebRTC and RTMP servers.

use std::net::SocketAddr;
use std::sync::Arc;

/// What a client asks to do on a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Play it.
    Play,
    /// Publish into it.
    Publish,
}

/// What a client presented to say who it is, from its `Authorization`
/// header.
#[derive(Clone, PartialEq, Eq)]
pub enum Credentials {
    /// Nothing, or a scheme the server does not read.
    None,
    /// `Bearer <token>` — what a WHIP or WHEP client sends.
    Bearer(String),
    /// `Basic` — what an RTSP player that cannot answer a Digest challenge
    /// sends for `rtsp://user:password@…`: the password itself, readable by
    /// anyone between them.
    Basic {
        /// The user name.
        user: String,
        /// The password.
        password: String,
    },
    /// `Digest` — what an RTSP player sends for `rtsp://user:password@…`
    /// where the server offers it, which an `RtspServer` does: the password
    /// itself never sent, so it is checked against the user's with
    /// [`RtspDigest::verifies`](crate::elements::RtspDigest::verifies).
    #[cfg(feature = "rtsp-server")]
    Digest(crate::elements::RtspDigest),
}

impl std::fmt::Debug for Credentials {
    /// Names the scheme and the user, never a secret, so that a request
    /// logged as it is keeps none.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => f.write_str("None"),
            Self::Bearer(_) => f.write_str("Bearer(..)"),
            Self::Basic { user, .. } => write!(f, "Basic {{ user: {user:?}, .. }}"),
            #[cfg(feature = "rtsp-server")]
            Self::Digest(digest) => write!(f, "Digest({digest:?})"),
        }
    }
}

// An RTMP server reads its publishers' credentials from a query, not a
// header.
#[cfg(any(feature = "rtsp-server", feature = "webrtc"))]
impl Credentials {
    /// What an `Authorization` header's value says, where there is one.
    pub(crate) fn from_header(value: Option<&str>) -> Self {
        let Some((scheme, rest)) = value.map(str::trim).and_then(|v| v.split_once(' ')) else {
            return Self::None;
        };
        let rest = rest.trim();
        if scheme.eq_ignore_ascii_case("bearer") && !rest.is_empty() {
            return Self::Bearer(rest.to_owned());
        }
        if scheme.eq_ignore_ascii_case("basic")
            && let Some(decoded) = base64_decode(rest)
            && let Ok(decoded) = String::from_utf8(decoded)
            && let Some((user, password)) = decoded.split_once(':')
        {
            return Self::Basic {
                user: user.to_owned(),
                password: password.to_owned(),
            };
        }
        Self::None
    }
}

/// One client's request to play or publish a path, as a server puts it to
/// its [`Authorize`].
#[derive(Debug, Clone)]
pub struct AccessRequest {
    /// The path, as it is served — no leading `/`.
    pub path: String,
    /// What the client asks to do on it.
    pub access: Access,
    /// What it presented.
    pub credentials: Credentials,
    /// Where it connects from.
    pub peer: SocketAddr,
}

/// Decides who may play or publish what: asked for each client that asks
/// to, before the server looks at anything else about the request, and
/// answering whether it may. A server without one lets everybody.
///
/// Refused, a client is answered `401 Unauthorized` — with
/// `WWW-Authenticate: Digest` and `Basic` over RTSP, which is what makes a
/// player given `rtsp://user:password@…` send them, hashed where it can, and
/// `Bearer` over HTTP. The decision is the application's alone: a list kept
/// in memory, a token's signature checked, a lookup. A Digest response
/// carries no password, so an application that lets users in by password
/// keeps each user's and checks it with `RtspDigest::verifies`.
///
/// It is called on a thread of the server's. A WebRTC server's HTTP
/// endpoint asks it on the request's own thread; an RTSP server asks it
/// on one of its runtime's, which it lets go of to other connections
/// first, so a slow answer — a database asked — holds up that client
/// alone. What it is handed is never logged with a secret in it.
///
/// Cheap to clone: clones are the one function.
///
/// # Example
///
/// ```
/// use media_pp::elements::{Access, Authorize, Credentials};
///
/// // Anybody plays; publishing takes the one token.
/// let authorize = Authorize::new(|request| match request.access {
///     Access::Play => true,
///     Access::Publish => request.credentials == Credentials::Bearer("s3cret".into()),
/// });
/// ```
///
/// Users with passwords, over RTSP — `rtsp-server` feature:
///
/// ```
/// # #[cfg(feature = "rtsp-server")] {
/// use media_pp::elements::{Authorize, Credentials};
///
/// fn password_of(user: &str) -> Option<&'static str> {
///     (user == "ana").then_some("s3cret")
/// }
///
/// let authorize = Authorize::new(|request| match &request.credentials {
///     Credentials::Digest(digest) => {
///         password_of(digest.user()).is_some_and(|password| digest.verifies(password))
///     }
///     Credentials::Basic { user, password } => password_of(user) == Some(password.as_str()),
///     _ => false,
/// });
/// # }
/// ```
#[derive(Clone)]
pub struct Authorize(Arc<dyn Fn(&AccessRequest) -> bool + Send + Sync>);

impl Authorize {
    /// Answers each request with what `decide` says.
    pub fn new(decide: impl Fn(&AccessRequest) -> bool + Send + Sync + 'static) -> Self {
        Self(Arc::new(decide))
    }

    pub(crate) fn allows(&self, request: &AccessRequest) -> bool {
        (self.0)(request)
    }
}

impl std::fmt::Debug for Authorize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Authorize(..)")
    }
}

/// Standard base64, padded or not; `None` for anything else.
#[cfg(any(feature = "rtsp-server", feature = "webrtc"))]
fn base64_decode(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let mut bits = 0u32;
    let mut count = 0;
    for byte in text.trim_end_matches('=').bytes() {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        };
        bits = (bits << 6) | u32::from(value);
        count += 6;
        if count >= 8 {
            count -= 8;
            out.push((bits >> count) as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(feature = "rtsp-server", feature = "webrtc"))]
    #[test]
    fn a_header_says_its_credentials() {
        assert_eq!(
            Credentials::from_header(Some("Bearer abc.def")),
            Credentials::Bearer("abc.def".into())
        );
        // "user:pa:ss" — a password may hold a colon; a user may not.
        assert_eq!(
            Credentials::from_header(Some("basic dXNlcjpwYTpzcw==")),
            Credentials::Basic {
                user: "user".into(),
                password: "pa:ss".into()
            }
        );
        assert_eq!(Credentials::from_header(None), Credentials::None);
        assert_eq!(Credentials::from_header(Some("Bearer")), Credentials::None);
        assert_eq!(
            Credentials::from_header(Some("Basic %%%")),
            Credentials::None
        );
        assert_eq!(
            Credentials::from_header(Some("Digest username=\"a\"")),
            Credentials::None
        );
    }

    #[test]
    fn credentials_shown_keep_their_secrets() {
        let shown = format!(
            "{:?} {:?}",
            Credentials::Bearer("s3cret".into()),
            Credentials::Basic {
                user: "ana".into(),
                password: "s3cret".into()
            }
        );
        assert!(!shown.contains("s3cret"), "{shown}");
        assert!(shown.contains("ana"), "{shown}");
    }
}
