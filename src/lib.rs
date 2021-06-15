#![feature(drain_filter)]
#![allow(dead_code)]
#![allow(unused_parens)]

mod functions;
mod types;

pub use self::functions::*;
pub use self::types::*;

#[allow(dead_code)]
const CACHE_LINE_SIZE: usize = 64;

#[allow(dead_code)]
type CacheLinePadding = ::core::mem::MaybeUninit<[u8; CACHE_LINE_SIZE]>;
