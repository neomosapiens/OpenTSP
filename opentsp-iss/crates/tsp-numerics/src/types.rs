//! Element types (`core/types.sail`).

/// An element type of a VXM instruction. The discriminant is the 5-bit type code
/// (`encdec_eltype`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EType {
    I8 = 0,
    F16 = 1,
    I32 = 2,
    F32 = 3,
    U8 = 4,
    I16 = 5,
    U16 = 6,
    U32 = 7,
    B8 = 8,
    B16 = 9,
    B32 = 10,
    /// FP8, 4 exponent and 3 fraction bits, no infinity.
    E4M3 = 11,
    /// FP8, 5 exponent and 2 fraction bits, IEEE style.
    E5M2 = 12,
    /// bfloat16, a profile extension (the type-extension bit of the code is set).
    Bf16 = 16,
}

impl EType {
    /// The thirteen types of the base set in code order (the position is the code).
    pub const ALL: [EType; 13] = [
        EType::I8,
        EType::F16,
        EType::I32,
        EType::F32,
        EType::U8,
        EType::I16,
        EType::U16,
        EType::U32,
        EType::B8,
        EType::B16,
        EType::B32,
        EType::E4M3,
        EType::E5M2,
    ];

    /// The extension types (a profile declares them in `extensions`), after the base set in [`EType::index`] order.
    pub const EXTENSIONS: [EType; 1] = [EType::Bf16];

    /// The type of a 5-bit code; `None` for the undefined codes 13..15 and 17..31. A profile that does not declare the
    /// extension of a type never decodes its code (`ots_isa::Isa::decode`).
    pub fn from_code(c: u8) -> Option<EType> {
        match c {
            0..=12 => EType::ALL.get(c as usize).copied(),
            16 => Some(EType::Bf16),
            _ => None,
        }
    }

    /// The position among the base set and then the extensions (the code for the base set, 13 for bf16).
    pub fn index(self) -> usize {
        match self {
            EType::Bf16 => 13,
            t => t as usize,
        }
    }

    /// The 5-bit type code (`encdec_eltype`).
    pub fn code(self) -> u8 {
        self as u8
    }

    /// Element width in bytes: 1, 2 or 4. // model: core/types.sail eltype_width
    pub fn width(self) -> u32 {
        match self {
            EType::I8 | EType::U8 | EType::B8 | EType::E4M3 | EType::E5M2 => 1,
            EType::F16 | EType::I16 | EType::U16 | EType::B16 | EType::Bf16 => 2,
            _ => 4,
        }
    }

    /// Element width in bits. // model: core/numerics/lanes.sail nbits_of
    pub fn nbits(self) -> u32 {
        8 * self.width()
    }

    /// `u8`, `u16` or `u32`. // model: core/types.sail is_unsigned
    pub fn is_unsigned(self) -> bool {
        matches!(self, EType::U8 | EType::U16 | EType::U32)
    }

    /// `b8`, `b16` or `b32`. // model: core/types.sail is_bool
    pub fn is_bool(self) -> bool {
        matches!(self, EType::B8 | EType::B16 | EType::B32)
    }

    /// `f16`, `f32`, `e4m3`, `e5m2` or `bf16`. // model: core/numerics/lanes.sail is_float
    pub fn is_float(self) -> bool {
        matches!(self, EType::F16 | EType::F32 | EType::E4M3 | EType::E5M2 | EType::Bf16)
    }

    /// The assembler name (`eltype_name`).
    pub fn name(self) -> &'static str {
        match self {
            EType::I8 => "i8",
            EType::F16 => "f16",
            EType::I32 => "i32",
            EType::F32 => "f32",
            EType::U8 => "u8",
            EType::I16 => "i16",
            EType::U16 => "u16",
            EType::U32 => "u32",
            EType::B8 => "b8",
            EType::B16 => "b16",
            EType::B32 => "b32",
            EType::E4M3 => "e4m3",
            EType::E5M2 => "e5m2",
            EType::Bf16 => "bf16",
        }
    }
}
