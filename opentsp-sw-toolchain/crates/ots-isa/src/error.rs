//! The error type of the assembler and its format layers.
use crate::Code;
use std::fmt;

/// A diagnostic with its code. `where_` is what precedes the code when printed
/// (`path:line`, `path:end`, `path` or the command name, as `tsp_as.py` prints it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsmError {
    pub code: Code,
    pub where_: Option<String>,
    pub msg: String,
}

impl AsmError {
    pub fn new(code: Code, msg: impl Into<String>) -> AsmError {
        AsmError { code, where_: None, msg: msg.into() }
    }

    pub fn at(mut self, where_: impl Into<String>) -> AsmError {
        self.where_ = Some(where_.into());
        self
    }

    /// `<where>: <code>: <message>`, the line `tsp_as.py` prints; `cmd` stands in for a missing `where_`.
    pub fn render(&self, cmd: &str) -> String {
        format!("{}: {}: {}", self.where_.as_deref().unwrap_or(cmd), self.code.as_str(), self.msg)
    }
}

impl fmt::Display for AsmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code.as_str(), self.msg)
    }
}

impl std::error::Error for AsmError {}
