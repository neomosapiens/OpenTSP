//! A model fault: the Sail model's `internal_error` (a state the specification says cannot occur).
//!
//! The numerics sit below the lane kernels, which return plain values, so a fault leaves them as an
//! unwinding panic with a [`ModelFault`] payload. `resume_unwind` does not run the panic hook, so a
//! library user sees no stray message; `tsp_iss::Machine` catches the payload at the cycle that
//! raised it and returns `RunError::Internal`.

/// The payload of a model fault: the message, without the cycle (the caller knows it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelFault(pub String);

/// Raise a model fault. // model: prelude/errors.sail internal_error
pub fn model_fault(msg: impl Into<String>) -> ! {
    std::panic::resume_unwind(Box::new(ModelFault(msg.into())))
}
