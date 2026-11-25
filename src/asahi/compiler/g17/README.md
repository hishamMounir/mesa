# G17 (Apple M5) machine-code layer

Table-driven encoder, decoder and instruction-length model for the G17 shader ISA, plus
`g17_pack_binary` (`g17_compile.c`), the G17 analogue of `agx_pack_binary`. The front-end
(NIR -> AGX IR, optimization, scheduling, RA) is shared; `agx_compile_function_nir` calls
`g17_pack_binary` when `key->dev.is_g17` is set.

Only a compute subset translates: moves, `thread_position_in_grid.x`, the integer and float ALU forms
in `g17_isa.py`, shifts and bitfield extract, device load/store, selects (lowered to an if/else) and
single-level loops. Anything else stops compilation with the opcode name. Nothing in Mesa sets
`is_g17` yet: `agx_open_device` refuses G17G until its compute launch is understood.

| File | Role |
|---|---|
| `g17_isa.py` | the ISA table: every instruction form as a byte template plus named bit-scattered fields |
| `g17_length.json` | instruction-length model measured on an M5 (ordered byte rules plus a two-level table) |
| `g17_gen.py` | generates `g17_isa.{h,c}` (op/field enums, descriptors, masks, length table) and `g17_golden.h` |
| `g17.h`, `g17_pack.c` | `g17_pack` / `g17_unpack` / `g17_length`, field access, back-edge offsets |
| `g17_compile.{c,h}` | `g17_pack_binary`: AGX IR -> G17 bytes (part of libasahi_compiler) |
| `test/g17_golden_vectors.json`, `test/g17_test.c` | kernels that ran on an M5 with the expected output; the test decodes and re-encodes them and checks back-edge targets and field round-trips |
| `test/test-g17-pack.cpp` | AGX IR -> G17 bytes, per case |

Model notes:
- Instructions are 2–18 bytes long.
- Registers r0..r63; every register field is 6 bits. Uniform word indices u0..u63.
- Discard (last-use) hint fields `discard_*`, and the performance-only `cache` hint.
- Compares write a predicate (`p`, `b0[7:5]`); branches and loop back-edges read one (`b1[7:4]`).
- Loop back-edges jump to `fe + 4 + 2 * offset` (signed halfwords).
- Forward branches find their targets structurally: `branch, then, jump, else, endif`.
- Device memory: `load`/`store` address `base pair + scale * r[idx] + 4 * offset`, with a component
  `mask`; the base pair is in the uniform or the general register file. Loads are asynchronous, so the
  first consumer sets `wait` (`store_w`/`load_w` b9.5, 27-family b0.3).
- `ustore` writes a uniform from the preamble; it has no effect from the main body.
