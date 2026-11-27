//! A small JSON reader that keeps what `serde_json::Value` loses: duplicate keys (in order) and
//! the literal text of numbers. The configuration checker needs both (`tsp_config_check.c`
//! reads numbers as decimal literals and refuses a duplicate key).

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// The literal text of the number.
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// First member of an object with this key (case sensitive).
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// Parse one JSON text. `Err(n)` is the byte offset of the error. The whole input must be one
/// value (only white space may follow).
pub fn parse(text: &[u8]) -> Result<Json, usize> {
    let mut p = Parser { s: text, i: 0 };
    p.ws();
    let v = p.value(0)?;
    p.ws();
    if p.i != text.len() {
        return Err(p.i);
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

const MAX_DEPTH: usize = 256;

impl Parser<'_> {
    fn ws(&mut self) {
        while matches!(self.s.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn lit(&mut self, word: &str, v: Json) -> Result<Json, usize> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(self.i)
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, usize> {
        if depth > MAX_DEPTH {
            return Err(self.i);
        }
        match self.peek().ok_or(self.i)? {
            b'n' => self.lit("null", Json::Null),
            b't' => self.lit("true", Json::Bool(true)),
            b'f' => self.lit("false", Json::Bool(false)),
            b'"' => Ok(Json::Str(self.string()?)),
            b'[' => {
                self.i += 1;
                let mut v = Vec::new();
                self.ws();
                if self.peek() == Some(b']') {
                    self.i += 1;
                    return Ok(Json::Arr(v));
                }
                loop {
                    self.ws();
                    v.push(self.value(depth + 1)?);
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(v));
                        }
                        _ => return Err(self.i),
                    }
                }
            }
            b'{' => {
                self.i += 1;
                let mut m = Vec::new();
                self.ws();
                if self.peek() == Some(b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(m));
                }
                loop {
                    self.ws();
                    if self.peek() != Some(b'"') {
                        return Err(self.i);
                    }
                    let k = self.string()?;
                    self.ws();
                    if self.peek() != Some(b':') {
                        return Err(self.i);
                    }
                    self.i += 1;
                    self.ws();
                    m.push((k, self.value(depth + 1)?));
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(m));
                        }
                        _ => return Err(self.i),
                    }
                }
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(self.i),
        }
    }

    fn number(&mut self) -> Result<Json, usize> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        let digits = |p: &mut Self| {
            let b = p.i;
            while matches!(p.peek(), Some(b'0'..=b'9')) {
                p.i += 1;
            }
            p.i > b
        };
        match self.peek() {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                digits(self);
            }
            _ => return Err(self.i),
        }
        if self.peek() == Some(b'.') {
            self.i += 1;
            if !digits(self) {
                return Err(self.i);
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if !digits(self) {
                return Err(self.i);
            }
        }
        Ok(Json::Num(String::from_utf8_lossy(&self.s[start..self.i]).into_owned()))
    }

    fn hex4(&mut self) -> Result<u32, usize> {
        let h = self.s.get(self.i..self.i + 4).ok_or(self.i)?;
        let t = std::str::from_utf8(h).map_err(|_| self.i)?;
        let v = u32::from_str_radix(t, 16).map_err(|_| self.i)?;
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, usize> {
        self.i += 1; // opening quote
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = self.peek().ok_or(self.i)?;
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).map_err(|_| self.i),
                0..=0x1f => return Err(self.i - 1),
                b'\\' => {
                    let e = self.peek().ok_or(self.i)?;
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let mut u = self.hex4()?;
                            if (0xD800..0xDC00).contains(&u) {
                                if self.s.get(self.i..self.i + 2) != Some(b"\\u") {
                                    return Err(self.i);
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.i);
                                }
                                u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
                            }
                            char::from_u32(u).ok_or(self.i)?
                        }
                        _ => return Err(self.i - 1),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                _ => out.push(c),
            }
        }
    }
}
