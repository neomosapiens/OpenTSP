//! The configuration checker's JSON reader: it keeps duplicate keys and the literal text of
//! numbers, and an error is the byte offset where the text stops being JSON.
use ots_profile::json::{parse, Json};

fn ok(t: &str) -> Json {
    parse(t.as_bytes()).unwrap_or_else(|at| panic!("{t:?}: error at {at}"))
}

fn err(t: &[u8]) -> usize {
    match parse(t) {
        Ok(v) => panic!("{:?} parsed as {v:?}", String::from_utf8_lossy(t)),
        Err(at) => at,
    }
}

fn s(t: &str) -> String {
    match ok(t) {
        Json::Str(s) => s,
        v => panic!("{t:?} is {v:?}"),
    }
}

#[test]
fn scalars_and_white_space() {
    assert_eq!(ok(" \t\r\nnull \n"), Json::Null);
    assert_eq!(ok("true"), Json::Bool(true));
    assert_eq!(ok("false"), Json::Bool(false));
    assert_eq!(ok("[]"), Json::Arr(vec![]));
    assert_eq!(ok("{ }"), Json::Obj(vec![]));
    assert_eq!(ok("[ 1 , [ ] , { } ]"), Json::Arr(vec![Json::Num("1".into()), Json::Arr(vec![]), Json::Obj(vec![])]));
}

#[test]
fn numbers_keep_their_literal_text() {
    for t in ["0", "-0", "7", "-12", "0.5", "10.25", "1e9", "1E+2", "-3e-04", "2.5E7"] {
        assert_eq!(ok(t), Json::Num(t.into()), "{t}");
    }
    // a leading zero ends the number; the digit after it is trailing garbage
    assert_eq!(err(b"01"), 1);
    assert_eq!(err(b"-"), 1);
    assert_eq!(err(b"-x"), 1);
    assert_eq!(err(b"1."), 2);
    assert_eq!(err(b"1.e3"), 2);
    assert_eq!(err(b"1e"), 2);
    assert_eq!(err(b"1e+"), 3);
    assert_eq!(err(b"+1"), 0);
}

#[test]
fn objects_keep_duplicate_keys_in_order() {
    let v = ok(r#"{"a": 1, "b": "x", "a": 2}"#);
    let Json::Obj(m) = &v else { panic!("{v:?}") };
    assert_eq!(m.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(), ["a", "b", "a"]);
    assert_eq!(v.get("a"), Some(&Json::Num("1".into())));
    assert_eq!(v.get("b").and_then(Json::as_str), Some("x"));
    assert_eq!(v.get("A"), None);
    assert_eq!(v.get("a").and_then(Json::as_str), None);
    assert_eq!(Json::Arr(vec![]).get("a"), None);
}

#[test]
fn string_escapes() {
    assert_eq!(s(r#""a\"\\\/\b\f\n\r\tz""#), "a\"\\/\u{8}\u{c}\n\r\tz");
    assert_eq!(s(r#""\u0041\u00e9\u20AC""#), "A\u{e9}\u{20ac}");
    assert_eq!(s(r#""\ud83d\ude00""#), "\u{1f600}");
    assert_eq!(s("\"h\u{e9}\""), "h\u{e9}");
}

#[test]
fn bad_strings_name_the_offending_byte() {
    assert_eq!(err(b"\"abc"), 4); // unterminated
    assert_eq!(err(b"\"a\nb\""), 2); // a control character
    assert_eq!(err(b"\"\\q\""), 2); // an unknown escape
    assert_eq!(err(b"\"\\"), 2); // the input ends inside an escape
    assert_eq!(err(b"\"\\u12\""), 3); // fewer than four hex digits
    assert_eq!(err(b"\"\\u12g4\""), 3);
    assert_eq!(err(b"\"\\u\xff\xff\xff\xff\""), 3); // not UTF-8 where the digits should be
    assert_eq!(err(b"\"\\ud800x\""), 7); // a high surrogate without a low one
    assert_eq!(err(b"\"\\ud800\\u0041\""), 13); // ... or with a non-surrogate after it
    assert_eq!(err(b"\"\\udc00\""), 7); // a lone low surrogate is not a character
    assert_eq!(err(b"\"\xff\""), 3); // the bytes are not UTF-8
}

#[test]
fn bad_structure_names_the_offending_byte() {
    assert_eq!(err(b""), 0);
    assert_eq!(err(b"   "), 3);
    assert_eq!(err(b"nul"), 0);
    assert_eq!(err(b"tru"), 0);
    assert_eq!(err(b"fals"), 0);
    assert_eq!(err(b"x"), 0);
    assert_eq!(err(b"[1 2]"), 3);
    assert_eq!(err(b"[1,"), 3);
    assert_eq!(err(b"[1,]"), 3);
    assert_eq!(err(b"{1:2}"), 1);
    assert_eq!(err(b"{\"a\" 2}"), 5);
    assert_eq!(err(b"{\"a\":2 \"b\":3}"), 7);
    assert_eq!(err(b"{\"a\":2,}"), 7);
    assert_eq!(err(b"{\"a\":}"), 5);
    assert_eq!(err(b"{\"a"), 3);
    assert_eq!(err(b"null x"), 5);
    assert_eq!(err(b"{} {}"), 3);
}

#[test]
fn nesting_is_bounded() {
    let deep = |n: usize| format!("{}{}", "[".repeat(n), "]".repeat(n));
    assert!(parse(deep(257).as_bytes()).is_ok());
    assert_eq!(err(deep(258).as_bytes()), 257);
    let objs = format!("{}1{}", "{\"k\":".repeat(258), "}".repeat(258));
    assert_eq!(err(objs.as_bytes()), 257 * 5);
}
