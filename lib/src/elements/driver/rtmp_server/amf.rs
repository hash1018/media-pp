//! AMF0, the encoding of RTMP's commands and metadata — as much of it as a
//! publisher sends and a server answers with.

/// One AMF0 value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Amf {
    Number(f64),
    Bool(bool),
    String(String),
    /// An object, or an ECMA array: named values, in order.
    Object(Vec<(String, Amf)>),
    Null,
    Undefined,
    StrictArray(Vec<Amf>),
}

impl Amf {
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }

    pub(crate) fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(number) => Some(*number),
            _ => None,
        }
    }

    /// The value named `name` of an object.
    pub(crate) fn get(&self, name: &str) -> Option<&Amf> {
        match self {
            Self::Object(fields) => fields
                .iter()
                .find(|(field, _)| field == name)
                .map(|(_, value)| value),
            _ => None,
        }
    }
}

/// How deep objects may nest before what is sent is taken to be malformed.
const MAX_DEPTH: usize = 16;

/// Every value in `data`, in order; `None` where it is malformed or uses a
/// type a publisher has no reason to send.
pub(crate) fn decode_all(mut data: &[u8]) -> Option<Vec<Amf>> {
    let mut values = Vec::new();
    while !data.is_empty() {
        values.push(decode(&mut data, 0)?);
    }
    Some(values)
}

fn take<'a>(data: &mut &'a [u8], count: usize) -> Option<&'a [u8]> {
    if data.len() < count {
        return None;
    }
    let (taken, rest) = data.split_at(count);
    *data = rest;
    Some(taken)
}

fn short_string(data: &mut &[u8]) -> Option<String> {
    let length = u16::from_be_bytes(take(data, 2)?.try_into().ok()?) as usize;
    Some(String::from_utf8_lossy(take(data, length)?).into_owned())
}

/// Named values up to the object end marker (an empty name and 9).
fn fields(data: &mut &[u8], depth: usize) -> Option<Vec<(String, Amf)>> {
    let mut fields = Vec::new();
    loop {
        let name = short_string(data)?;
        if name.is_empty() && data.first() == Some(&9) {
            *data = &data[1..];
            return Some(fields);
        }
        fields.push((name, decode(data, depth + 1)?));
    }
}

fn decode(data: &mut &[u8], depth: usize) -> Option<Amf> {
    if depth > MAX_DEPTH {
        return None;
    }
    let marker = take(data, 1)?[0];
    Some(match marker {
        0 => Amf::Number(f64::from_be_bytes(take(data, 8)?.try_into().ok()?)),
        1 => Amf::Bool(take(data, 1)?[0] != 0),
        2 => Amf::String(short_string(data)?),
        3 => Amf::Object(fields(data, depth)?),
        5 => Amf::Null,
        6 => Amf::Undefined,
        8 => {
            // An ECMA array's count is a hint; its end marker ends it.
            take(data, 4)?;
            Amf::Object(fields(data, depth)?)
        }
        10 => {
            let count = u32::from_be_bytes(take(data, 4)?.try_into().ok()?) as usize;
            let mut items = Vec::with_capacity(count.min(64));
            for _ in 0..count {
                items.push(decode(data, depth + 1)?);
            }
            Amf::StrictArray(items)
        }
        11 => {
            // A date: milliseconds and a time zone no one reads.
            let millis = f64::from_be_bytes(take(data, 8)?.try_into().ok()?);
            take(data, 2)?;
            Amf::Number(millis)
        }
        12 => {
            let length = u32::from_be_bytes(take(data, 4)?.try_into().ok()?) as usize;
            Amf::String(String::from_utf8_lossy(take(data, length)?).into_owned())
        }
        _ => return None,
    })
}

/// `values`, encoded in order.
pub(crate) fn encode_all(values: &[Amf]) -> Vec<u8> {
    let mut out = Vec::new();
    for value in values {
        encode(value, &mut out);
    }
    out
}

fn encode_name(name: &str, out: &mut Vec<u8>) {
    let bytes = &name.as_bytes()[..name.len().min(u16::MAX as usize)];
    out.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
    out.extend_from_slice(bytes);
}

fn encode(value: &Amf, out: &mut Vec<u8>) {
    match value {
        Amf::Number(number) => {
            out.push(0);
            out.extend_from_slice(&number.to_be_bytes());
        }
        Amf::Bool(value) => out.extend_from_slice(&[1, u8::from(*value)]),
        Amf::String(text) if text.len() <= u16::MAX as usize => {
            out.push(2);
            encode_name(text, out);
        }
        Amf::String(text) => {
            out.push(12);
            out.extend_from_slice(&(text.len() as u32).to_be_bytes());
            out.extend_from_slice(text.as_bytes());
        }
        Amf::Object(fields) => {
            out.push(3);
            for (name, value) in fields {
                encode_name(name, out);
                encode(value, out);
            }
            out.extend_from_slice(&[0, 0, 9]);
        }
        Amf::Null => out.push(5),
        Amf::Undefined => out.push(6),
        Amf::StrictArray(items) => {
            out.push(10);
            out.extend_from_slice(&(items.len() as u32).to_be_bytes());
            for item in items {
                encode(item, out);
            }
        }
    }
}

/// An object of `fields`, for writing.
pub(crate) fn object(fields: &[(&str, Amf)]) -> Amf {
    Amf::Object(
        fields
            .iter()
            .map(|(name, value)| ((*name).to_owned(), value.clone()))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_is_encoded_decodes_the_same() {
        let values = vec![
            Amf::String("connect".into()),
            Amf::Number(1.0),
            object(&[
                ("app", Amf::String("live".into())),
                ("fpad", Amf::Bool(false)),
                ("nested", object(&[("level", Amf::Number(2.5))])),
            ]),
            Amf::Null,
            Amf::Undefined,
            Amf::StrictArray(vec![Amf::Number(1.0), Amf::String("a".into())]),
        ];
        assert_eq!(decode_all(&encode_all(&values)), Some(values));
    }

    #[test]
    fn an_ecma_array_reads_as_an_object() {
        // @setDataFrame's onMetaData as FFmpeg writes it: an ECMA array.
        let mut data = vec![8, 0, 0, 0, 1];
        encode_name("width", &mut data);
        encode(&Amf::Number(1920.0), &mut data);
        data.extend_from_slice(&[0, 0, 9]);
        let values = decode_all(&data).unwrap();
        assert_eq!(values[0].get("width"), Some(&Amf::Number(1920.0)));
    }

    #[test]
    fn what_is_malformed_is_none() {
        assert_eq!(decode_all(&[2, 0, 9, b'a']), None, "a string cut short");
        assert_eq!(decode_all(&[0x7f]), None, "an unknown type");
        // Objects nested past the bound.
        let mut deep = Vec::new();
        for _ in 0..40 {
            deep.extend_from_slice(&[3, 0, 1, b'a']);
        }
        assert_eq!(decode_all(&deep), None);
    }
}
