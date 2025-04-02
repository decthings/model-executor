#![allow(clippy::type_complexity)]

#[cfg(unix)]
pub mod unix;

#[cfg(feature = "wasm")]
pub mod wasm;
