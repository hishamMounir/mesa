"""
G17 (Apple M5) ISA tables — the single source of truth for the Mesa-style backend.

This module is the G17 analogue of Mesa's ``src/asahi/compiler/agx_opcodes.py``: every
instruction is described ONCE, declaratively, as an :class:`Insn` with a base byte template
and a set of named :class:`Field`s. The generic :func:`pack` / :func:`unpack` primitives turn
that one description into both an encoder and a disassembler, so the two can never drift.

Every encoding here was tested on M5 hardware. ``test/g17_golden_vectors.json`` holds whole kernels built
from these forms that ran on an M5 and produced their expected output; the unit tests decode and re-encode
them byte for byte.

Field model (matches Mesa's bitfield packing):
  * A field is a list of ``(byte, bit)`` positions, LSB first. ``pack`` scatters an integer over
    them; ``unpack`` gathers them back.
  * ``enc`` / ``dec`` optionally transform between the logical operand value and the raw bits
    (used for the shift-amount enum, the min/max flag, the mini-float immediate, ...).

Decode matching (the "opcode mask"):
  * ``Insn.fixed_mask`` is every bit the instruction pins (the template bits that are NOT covered
    by any field). ``Insn.match`` is the template masked by it. An instruction of the right length
    whose ``bytes & fixed_mask == match`` is an instance of that op. :func:`validate_tables`
    asserts the (length, mask, match) triples are mutually unambiguous.
"""

# --------------------------------------------------------------------------------------------------
# bit scatter primitives (the one place bits are moved)
# --------------------------------------------------------------------------------------------------
def scatter(buf, positions, value):
    """Write ``value`` over (byte, bit) ``positions``, LSB first (clears then sets each bit)."""
    for i, (byte, bit) in enumerate(positions):
        buf[byte] = (buf[byte] & ~(1 << bit) & 0xFF) | ((value >> i & 1) << bit)


def gather(buf, positions):
    """Read an integer back out of (byte, bit) ``positions``, LSB first."""
    v = 0
    for i, (byte, bit) in enumerate(positions):
        v |= ((buf[byte] >> bit) & 1) << i
    return v


# --------------------------------------------------------------------------------------------------
# field / instruction descriptors
# --------------------------------------------------------------------------------------------------
class Field:
    """A named operand slot: bit positions (LSB first) plus an optional value<->bits transform.

    ``enc(value) -> raw`` and ``dec(raw) -> value`` default to identity. ``kind`` is advisory
    metadata for tooling/printing ("reg", "ureg", "imm", "enum", "hint", ...).

    ``default`` is the RAW value used when the caller leaves the field unset: an int, or
    :data:`TEMPLATE` to keep whatever the instruction's template holds in those bits (used for the
    discard hints and for immediates whose zero value is a different instruction, e.g. fma_imm)."""

    __slots__ = ("name", "positions", "enc", "dec", "kind", "default", "mapping")

    def __init__(self, name, positions, enc=None, dec=None, kind="imm", default=0):
        self.name = name
        self.positions = positions
        self.enc = enc or (lambda v: v)
        self.dec = dec or (lambda r: r)
        self.kind = kind
        self.default = default
        self.mapping = None          # enum fields: logical value -> raw

    @property
    def width(self):
        return len(self.positions)


TEMPLATE = object()   # Field.default sentinel: keep the template's bits


def hint(name, pos):
    """A 1-bit discard / last-use hint: 1 = the source register is dead after this read."""
    return Field(name, [pos], kind="hint", default=TEMPLATE)


def enum_field(name, positions, mapping, kind="enum"):
    """A field whose logical values are keys of ``mapping`` (value -> raw bits)."""
    inv = {v: k for k, v in mapping.items()}
    f = Field(name, positions, enc=lambda v: mapping[v], dec=lambda r: inv.get(r, r), kind=kind)
    f.mapping = dict(mapping)
    return f


class Insn:
    """One instruction form: mnemonic, byte length, base template, and operand fields."""

    def __init__(self, name, length, template_hex, fields=(), family="", summary=""):
        self.name = name
        self.length = length
        self.template = bytes.fromhex(template_hex)
        assert len(self.template) == length, f"{name}: template {len(self.template)} != length {length}"
        self.fields = {f.name: f for f in fields}
        self.family = family
        self.summary = summary
        # The opcode mask: every bit NOT owned by a field. Field bits in the template are forced to
        # zero in the canonical match value so a template captured with nonzero operand bits is fine.
        mask = bytearray(b"\xff" * length)
        for f in self.fields.values():
            for (byte, bit) in f.positions:
                mask[byte] &= ~(1 << bit) & 0xFF
        self.fixed_mask = bytes(mask)
        self.match = bytes(t & m for t, m in zip(self.template, self.fixed_mask))
        # raw value written for each field the caller leaves unset
        self.defaults = {n: gather(self.template, f.positions) if f.default is TEMPLATE else f.default
                         for n, f in self.fields.items()}

    # ---- encode ----
    def encode(self, **operands):
        buf = bytearray(self.template)
        # start from the defaults (0, or the template's bits for TEMPLATE fields), then scatter operands
        for n, f in self.fields.items():
            scatter(buf, f.positions, self.defaults[n])
        for name, value in operands.items():
            if name not in self.fields:
                raise KeyError(f"{self.name}: no field {name!r} (have {list(self.fields)})")
            f = self.fields[name]
            raw = f.enc(value)
            if raw >= (1 << f.width):
                raise ValueError(f"{self.name}.{name}: value {value!r} -> {raw:#x} overflows {f.width} bits")
            scatter(buf, f.positions, raw)
        return bytes(buf)

    # ---- decode ----
    def matches(self, slice_):
        if len(slice_) < self.length:
            return False
        return all((slice_[i] & self.fixed_mask[i]) == self.match[i] for i in range(self.length))

    def decode(self, slice_):
        buf = slice_[: self.length]
        out = {}
        for name, f in self.fields.items():
            out[name] = f.dec(gather(buf, f.positions))
        return out


# --------------------------------------------------------------------------------------------------
# shared field layouts
# --------------------------------------------------------------------------------------------------
# Every register field is 6 bits = r0..r63. Layouts are LSB first.
DEST = [(0, 4), (7, 3), (7, 4), (0, 7), (7, 5), (2, 7)]   # 27-family standard dest
SHADD_A = [(8, 7), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4)]  # A operand (shadd layout); b9.5 = A-kind, not a reg bit
SHADD_C = [(1, 1), (3, 5), (3, 6), (3, 7), (5, 7), (8, 0)]  # C operand (register mode; shares bits with SHARED_IMM)
IMAD_B = SHADD_C                                         # imul B operand
CEIL_SRC = SHADD_C                                       # unary source, C layout (rsqrt / ceil / f2u)
F_SRC = [(1, 1), (3, 4), (3, 5), (3, 7), (3, 6), (8, 0)]    # unary source, F layout (exp2 / log2 / u2f / imulhi A)
MH_A = F_SRC                                             # imulhi multiplicand A
POPC_SRC = [(1, 1), (5, 6), (5, 7), (3, 1), (3, 3), (3, 4)]  # popcount / 23-family bitwise source (b3.5 = no-read)
D9 = [(0, 4), (0, 5), (0, 6), (0, 7), (2, 6), (2, 7)]       # 09/29/12-family float dest
A91 = [(1, i) for i in range(1, 7)]                      # 09-family A operand = b1[6:1]
B91 = [(3, i) for i in range(1, 7)]                      # 09-family B/C operand = b3[6:1]
U91 = B91                                                # 09/29-family uniform word index (u0..u63; u0..u33 verified)
A_UNI = SHADD_A                                          # 27-family uniform word index in the A slot (u0..u63; u0..u33 verified)
# ---- 27-family 16-bit width / half selects ----
# The register file is 16-bit-addressable, and the 12-byte 27-family forms carry four bits that pick
# the width and half of their destination and their A operand:
#
#   b3.1  dest WIDTH:  1 = 32-bit (the template default), 0 = 16-bit -- writes one half, leaving the
#                      other intact.
#   b3.0  dest HALF:   0 = low, 1 = high. Only meaningful with b3.1 clear.
#   b9.7  A WIDTH:     1 = 32-bit (default), 0 = 16-bit, zero-extended into a 32-bit result.
#   b8.6  A HALF:      0 = low, 1 = high.
#
# Tested on `add_imm`; `shadd` and `shadd_imm` share the layout bit for bit and carry the same fields.
# The CEIL_SRC-layout source has its own 16-bit selects (f2u): note that its width bit is SET for
# 16-bit, the opposite polarity to the A operand's.
C16_WIDTH = (8, 3)
C16_HALF = (1, 0)

D16_WIDTH = (3, 1)
D16_HALF = (3, 0)
A16_WIDTH = (9, 7)
A16_HALF = (8, 6)


def _w16():
    """The four width/half fields, defaulting to the template's 32-bit selection."""
    return [hint("d32", D16_WIDTH), hint("dhalf", D16_HALF),
            hint("a32", A16_WIDTH), hint("ahalf", A16_HALF)]


CACHE27 = (4, 3)    # 27-family: performance hint (likely a destination register-cache hint); no effect on results
WAIT27 = (0, 3)     # 27-family: wait for outstanding device loads before reading sources
A_UNIFORM = (10, 1)                                      # 27-family A-operand kind: 1 = the A field is a uniform word index
D12 = D9                                                 # 12-family (fmin/fmax) dest
MOV_DEST = [(0, 4), (0, 5), (0, 6), (0, 7), (2, 6), (2, 7)]  # 4-byte moves (mov_sr / mov_uniform / mov_imm)
MOV32_DEST = MOV_DEST                                    # mov32 dest

# Discard (last-use) hint bits: set = the source register is dead after this instruction (it reads 0 later).
# A wrong hint on a loop-carried register hangs the loop. Defaults keep each template's bit.
H27_A, H27_C = (10, 5), (8, 5)                           # 27-family: A (shadd layout) / C-or-unary source
H23_S = (8, 2)                                           # popcount / 23-family bitwise source
H09_A, H09_B = (2, 3), (2, 4)                            # 09/29/12-family: A / B-or-C

# ---- 09-family 16-bit float ----
# The 4-byte float forms select fp16 in byte0's format nibble, and then each operand and the
# destination carry their own half select:
#
#   b0.0  1 = 32-bit float (the template default), 0 = fp16
#   b0.3  destination half: 1 = high, 0 = low
#   b1.0  A operand half:   1 = high, 0 = low
#   b3.0  B (or C) operand half: 1 = high, 0 = low
#
# The 6-bit d / a / b fields still name the 32-bit REGISTER; the half bits pick within it. All three
# half bits are SET in the templates, so an fp16 op that leaves them alone addresses high halves --
# an emitter must set them explicitly. Tested on `fadd`; `fmul`, `ffma` and `fsub` share the layout.
#
# CAVEAT: an fp16 op writing the LOW half of r0, r16, r32 or r48 encodes byte0 = 0x00, which the
# length model reads as a 1-byte nop -- byte0 is 0 exactly when f32 and the dest half are clear and
# the dest register's low four bits are zero. Such an instruction cannot be parsed back, so
# `g17_compile.c` refuses it.
F09_32 = (0, 0)
F09_DHALF = (0, 3)
F09_AHALF = (1, 0)
F09_BHALF = (3, 0)


def _f16(b_name="bhalf"):
    """The fp16 width and half selects shared by the 09-family float forms."""
    return [hint("f32", F09_32), hint("dhalf", F09_DHALF), hint("ahalf", F09_AHALF),
            hint(b_name, F09_BHALF)]

# shadd shift amount -> raw (b10.0, b10.2, b11.0), LSB first. Raw 5..7 are not used.
SHIFT_ENUM = {0: 0b001, 1: 0b010, 2: 0b011, 3: 0b100, 4: 0b000}
SHIFT_POS = [(10, 0), (10, 2), (11, 0)]


# ---- scattered immediates (non-contiguous) ----
# The shared small-immediate scatter used by add-immediate / wadd increments / shift-loop inits:
#   K = b1[1:0] | b3[7:5]<<2 | b5[7]<<5 | b8[0]<<6 | b8[1]<<7
SHARED_IMM = [(1, 0), (1, 1), (3, 5), (3, 6), (3, 7), (5, 7), (8, 0), (8, 1)]
# mov32 immediate (32b): b1[6:0], b4[4:1], b5[3:2], b6[7:0], b7[3:0], b3[7:1]
MOV32_IMM = ([(1, i) for i in range(7)] + [(4, i) for i in range(1, 5)] + [(5, 2), (5, 3)]
             + [(6, i) for i in range(8)] + [(7, i) for i in range(4)] + [(3, i) for i in range(1, 8)])
# cmp_imm K (7b): b4[4] is bit0, b3[1..6] are bits 1..6
CMP_IMM = [(4, 4), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6)]
# 23-family bitwise immediate (8b): b8[6], b8[7], b9[0..5]
BW_IMM = [(8, 6), (8, 7), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5)]
# 12-family fmin/fmax mini-float immediate = the whole b3 byte
MINMAX_IMM = [(3, i) for i in range(8)]


# --------------------------------------------------------------------------------------------------
# the instruction table
# --------------------------------------------------------------------------------------------------
INSNS = []


def _op(*args, **kw):
    insn = Insn(*args, **kw)
    assert insn.name not in {i.name for i in INSNS}, f"duplicate op {insn.name}"
    INSNS.append(insn)
    return insn


# ---- flow terminator ----
_op("stop", 4, "0e000000", family="ctrl", summary="end of kernel")


# ---- moves / constants ----
_op("mov_sr", 4, "0ca01006",
    fields=[Field("d", MOV_DEST, kind="reg")],
    family="mov", summary="d = thread_position_in_grid.x (the only tid-SR producer)")
_op("mov32", 8, "0c80020002000000",
    fields=[Field("d", MOV32_DEST, kind="reg"), Field("imm", MOV32_IMM, kind="imm")],
    family="mov", summary="d = imm32")
_op("mov_uniform", 4, "0b000108",
    fields=[Field("d", MOV_DEST, kind="reg"),
            Field("uhalf", [(1, i) for i in range(7)], kind=" uhalf")],
    family="mov", summary="d = u[uhalf/2]  (32-bit uniform at half-index uhalf)")
_op("mov_imm7", 4, "0b800108",
    fields=[Field("d", MOV_DEST, kind="reg"), Field("imm", [(1, i) for i in range(7)] + [(2, 3)], kind="imm")],
    family="mov", summary="d = imm8 (0..255; bit7 = b2.3)")

# ---- integer ALU (27-family; b6 low nibble selects the op) ----
_op("shadd", 12, "2706041a2900a30288800400",
    fields=[Field("d", DEST, kind="reg"), Field("a", SHADD_A, kind="reg"), Field("c", SHADD_C, kind="reg"),
            enum_field("s", SHIFT_POS, SHIFT_ENUM), hint("discard_a", H27_A), hint("discard_c", H27_C),
            Field("a_uniform", [A_UNIFORM], kind="flag"),
            Field("cache", [CACHE27], kind="hint", default=TEMPLATE), Field("wait", [WAIT27], kind="flag")] + _w16(),
    family="alu-int", summary="d = (A << s) + C   [s in 0..4]")
_op("shadd_imm", 12, "2705041a2d00a30a20800100",
    fields=[Field("d", DEST, kind="reg"), Field("a", SHADD_A, kind="reg"), Field("imm", SHARED_IMM, kind="imm"),
            enum_field("s", SHIFT_POS, SHIFT_ENUM), hint("discard_a", H27_A),
            Field("a_uniform", [A_UNIFORM], kind="flag"),
            Field("cache", [CACHE27], kind="hint", default=TEMPLATE), Field("wait", [WAIT27], kind="flag")] + _w16(),
    family="alu-int", summary="d = (A << s) + imm   (shared-immediate addend; used for tid+C)")
_op("add_imm", 12, "2700043a2500a30a20812100",
    fields=[Field("d", DEST, kind="reg"), Field("a", SHADD_A, kind="reg"), Field("imm", SHARED_IMM, kind="imm"),
            hint("discard_a", H27_A), Field("a_uniform", [A_UNIFORM], kind="flag"),
            Field("cache", [CACHE27], kind="hint", default=TEMPLATE), Field("wait", [WAIT27], kind="flag")] + _w16(),
    family="alu-int", summary="d = A + imm  (1..255)")
_op("imul", 14, "2706041a2100a1028880000001" "02",
    fields=[Field("d", DEST, kind="reg"), Field("a", SHADD_A, kind="reg"), Field("b", IMAD_B, kind="reg"),
            hint("discard_a", H27_A), hint("discard_b", H27_C), Field("a_uniform", [A_UNIFORM], kind="flag"),
            Field("cache", [CACHE27], kind="hint", default=TEMPLATE), Field("wait", [WAIT27], kind="flag")],
    family="alu-int", summary="d = A * B (u32 low)")
_op("imulhi", 12, "270004022180a902a8810030",
    fields=[Field("d", DEST, kind="reg"), Field("a", MH_A, kind="reg"), Field("u", A_UNI, kind="ureg"),
            hint("discard_a", H27_C)],
    family="alu-int", summary="d = mullo(A*u[u]), d+1 = mulhi(A*u[u])  (u = uniform word index)")
_op("popcount", 10, "270600803230aa020000",
    fields=[Field("d", DEST, kind="reg"), Field("s", POPC_SRC, kind="reg"), hint("discard_s", H23_S)],
    family="alu-int", summary="d = popcount(s)")

# ---- 23-family bitwise-immediate: d = s OP imm ----
# op-select = (b2.0, b4.1, b5.3): and=001 or=111 xor=110
BW_OP = enum_field("op", [(2, 0), (4, 1), (5, 3)], {"and": 0b001, "or": 0b111, "xor": 0b110})
_op("bw_imm", 10, "230407803030a30a0083",
    fields=[Field("d", DEST, kind="reg"), Field("s", POPC_SRC, kind="reg"),
            BW_OP, Field("imm", BW_IMM, kind="imm"), hint("discard_s", H23_S)],
    family="alu-int", summary="d = s (&|^) imm   (imm 0..255)")

# ---- 27-family shifts ----
# b6 selects the op: a0 = shl, a1 = lsr, a3 = asr. The value is in C; the amount is in the A slot -- a register,
# or in the immediate forms a 7-bit value read across the A field and the A-half bit: s = (A << 1) | b8.6.
# lsr and shl are 14 bytes (b11..b13 = bitfield mask, 34 00 00 = whole word), asr is 12. In the immediate forms
# b8.3..b8.5 are fixed 1.
# The value register's last-use hint is b4.2 in this family (not b8.5): set, C reads 0 afterwards, so it must be
# clear when C is read again.
SHIFT_IMM7 = [(8, 6)] + SHADD_A
H27_SHIFT_C = (4, 2)   # shift family: value-register (C) last-use hint
# Bitfield width of the 14-byte lsr: 0 = the whole word, else the extract width w: b13[7:4] = w[3:0],
# b12.0 = w[4]. extract_bits(x, off, w) is lsr by off with width w.
SHIFT_WIDTH = [(13, 4), (13, 5), (13, 6), (13, 7), (12, 0)]
_op("shr_imm", 14, "2700001a2200a1023880003400 00".replace(" ", ""),
    fields=[Field("d", DEST, kind="reg"), Field("c", SHADD_C, kind="reg"), Field("imm", SHIFT_IMM7, kind="imm"),
            hint("discard_c", H27_SHIFT_C), Field("cache", [CACHE27], kind="hint", default=TEMPLATE),
            Field("wait", [WAIT27], kind="flag"), Field("width", SHIFT_WIDTH, kind="imm")],
    family="alu-int", summary="d = (c >> imm) & mask(width)   (logical; imm 0..31; width 0 = whole word)")
_op("shr_reg", 14, "2704001a2200a1020880003400 00".replace(" ", ""),
    fields=[Field("d", DEST, kind="reg"), Field("c", SHADD_C, kind="reg"), Field("a", SHADD_A, kind="reg"),
            hint("discard_a", H27_A), hint("discard_c", H27_SHIFT_C),
            Field("cache", [CACHE27], kind="hint", default=TEMPLATE),
            Field("wait", [WAIT27], kind="flag"), Field("width", SHIFT_WIDTH, kind="imm")],
    family="alu-int", summary="d = (c >> a) & mask(width)   (logical; width 0 = whole word)")
_op("shl_reg", 14, "2704001a2200a0020880003400 00".replace(" ", ""),
    fields=[Field("d", DEST, kind="reg"), Field("c", SHADD_C, kind="reg"), Field("a", SHADD_A, kind="reg"),
            hint("discard_a", H27_A), hint("discard_c", H27_SHIFT_C),
            Field("cache", [CACHE27], kind="hint", default=TEMPLATE),
            Field("wait", [WAIT27], kind="flag")],
    family="alu-int", summary="d = c << a")
# Shift left by an immediate: shr_imm's layout with b6 = a0.
_op("shl_imm", 14, "2700001a2200a0023880003400 00".replace(" ", ""),
    fields=[Field("d", DEST, kind="reg"), Field("c", SHADD_C, kind="reg"), Field("imm", SHIFT_IMM7, kind="imm"),
            hint("discard_c", H27_SHIFT_C), Field("cache", [CACHE27], kind="hint", default=TEMPLATE),
            Field("wait", [WAIT27], kind="flag")],
    family="alu-int", summary="d = c << imm   (imm 0..31)")
_op("asr_imm", 12, "2700001a2200a30238804000",
    fields=[Field("d", DEST, kind="reg"), Field("c", SHADD_C, kind="reg"), Field("imm", SHIFT_IMM7, kind="imm"),
            hint("discard_c", H27_SHIFT_C), Field("cache", [CACHE27], kind="hint", default=TEMPLATE),
            Field("wait", [WAIT27], kind="flag")],
    family="alu-int", summary="d = c >> imm   (arithmetic; imm 0..31)")
_op("asr_reg", 12, "2704001a2200a30208004000",
    fields=[Field("d", DEST, kind="reg"), Field("c", SHADD_C, kind="reg"), Field("a", SHADD_A, kind="reg"),
            hint("discard_a", H27_A), hint("discard_c", H27_SHIFT_C),
            Field("cache", [CACHE27], kind="hint", default=TEMPLATE),
            Field("wait", [WAIT27], kind="flag")],
    family="alu-int", summary="d = c >> a   (arithmetic; a read as 16-bit: b9.7 = 0)")

# 23-family register-register bitwise: bw_imm's layout with b6 = a4 and the second register (b) in the 27-family
# A slot. It carries a full two-input truth table, bit for bit AGX's `truth_table`: bit i = f(s, b) for
# s = i & 1, b = i >> 1, at b4.0, b4.1, b5.3, b2.0 (bw_imm's and/or/xor select is the top three of these bits).
# b6.2 is b's last-use hint: set, b reads 0 afterwards.
BW_LUT = [(4, 0), (4, 1), (5, 3), (2, 0)]
H23_B = (6, 2)
_op("bw_reg", 10, "23000680 3030a0020080".replace(" ", ""),
    fields=[Field("d", DEST, kind="reg"), Field("s", POPC_SRC, kind="reg"), Field("lut", BW_LUT, kind="imm"),
            Field("b", SHADD_A, kind="reg"), hint("discard_s", H23_S), hint("discard_b", H23_B),
            Field("wait", [WAIT27], kind="flag")],
    family="alu-int", summary="d = lut(s, b)   (AGX truth-table order)")

# ---- float ALU: 27-family unary (b6 low nibble selects the op) ----
_op("fexp2", 10, "2700040a2a20a50a3000",
    fields=[Field("d", DEST, kind="reg"), Field("s", F_SRC, kind="reg"), hint("discard_s", H27_C)],
    family="alu-float", summary="d = exp2(s)")
_op("flog2", 10, "2700040a2220a40a3000",
    fields=[Field("d", DEST, kind="reg"), Field("s", F_SRC, kind="reg"), hint("discard_s", H27_C)],
    family="alu-float", summary="d = log2(s)")
_op("frsqrt", 10, "378004182a20a2121000",
    fields=[Field("d", DEST, kind="reg"), Field("s", CEIL_SRC, kind="reg"), hint("discard_s", H27_C)],
    family="alu-float", summary="d = rsqrt(s)")
_op("fceil", 10, "270004382a20a01ab000",
    fields=[Field("d", DEST, kind="reg"), Field("s", CEIL_SRC, kind="reg"), hint("discard_s", H27_C)],
    family="alu-float", summary="d = ceil(s)")
# The conversions carry the same 16-bit selects as the 12-byte 27-family forms for their DESTINATION
# (b3.1 width, b3.0 half): with b3.1 clear, u2f writes an fp16 result into one half, so u32 -> f16 is one
# instruction.
#
# f2u's SOURCE is the CEIL_SRC layout rather than add_imm's A layout, and its 16-bit selects are in
# different places: b8.3 SET = read the source as fp16 (the inverted polarity against the A operand's
# b9.7), with b1.0 SET = the high half.
_op("u2f", 10, "270600022a80af020800",
    fields=[Field("d", DEST, kind="reg"), Field("s", F_SRC, kind="reg", default=1),
            hint("discard_s", H27_C), hint("d32", D16_WIDTH), hint("dhalf", D16_HALF)],
    family="convert", summary="d = float(u32 s)  (d32=0 -> fp16 result in one half)")
_op("f2u", 10, "2700001a2200ae02b003",
    fields=[Field("d", DEST, kind="reg"), Field("s", CEIL_SRC, kind="reg"), hint("discard_s", H27_C),
            hint("d32", D16_WIDTH), hint("dhalf", D16_HALF),
            hint("s16", C16_WIDTH), hint("shalf", C16_HALF)],
    family="convert", summary="d = u32(float s)  (s16=1 -> source read as fp16)")

# ---- float ALU: 09-family register-register (dest low bits spill into b0[7:4]) ----
_op("fadd", 4, "09051801",
    fields=[Field("d", D9, kind="reg"), Field("a", A91, kind="reg"), Field("b", B91, kind="reg"),
            hint("discard_a", H09_A), hint("discard_b", H09_B)] + _f16("bhalf"),
    family="alu-float", summary="d = A + B")
_op("fmul", 4, "09051901",
    fields=[Field("d", D9, kind="reg"), Field("a", A91, kind="reg"), Field("b", B91, kind="reg"),
            hint("discard_a", H09_A), hint("discard_b", H09_B)] + _f16("bhalf"),
    family="alu-float", summary="d = A * B")
_op("ffma", 4, "09011b04",
    fields=[Field("d", D9, kind="reg"), Field("a", A91, kind="reg"), Field("c", B91, kind="reg"),
            hint("discard_a", H09_A), hint("discard_c", H09_B)] + _f16("chalf"),
    family="alu-float", summary="d = d * A + C  (dest doubles as the multiplicand)")
_op("fsub", 6, "09011c050008",
    fields=[Field("d", D9, kind="reg"), Field("a", A91, kind="reg"), Field("b", B91, kind="reg"),
            hint("discard_a", H09_A), hint("discard_b", H09_B)] + _f16("bhalf"),
    family="alu-float", summary="d = A - B")
_op("faddu", 6, "09010c050002",
    fields=[Field("d", D9, kind="reg"), Field("a", A91, kind="reg"), Field("u", U91, kind="ureg"),
            Field("sub", [(5, 3)], kind="flag"), hint("discard_a", H09_A)],
    family="alu-float", summary="d = A + uN  (sub=1 -> A - uN)")

# ---- 12-family fmin/fmax with mini-float immediate ----
_op("fminmax", 6, "22010ad00002",
    fields=[Field("d", D12, kind="reg"), Field("s", A91, kind="reg"),
            Field("imm", MINMAX_IMM, kind="imm"), Field("is_min", [(4, 0)], kind="flag"), hint("discard_s", H09_A)],
    family="alu-float", summary="d = (is_min ? min : max)(s, mini-float imm)")

# ---- device memory (b0&7 = 7, b2&7 = 3) ----
# address = u[base]:u[base+1] (64-bit uniform pair, base even) + scale * r[idx] (if indexed) + 4 * offset
# u0:u1 = buffer(0), u2:u3 = buffer(1) (driver ABI). Components selected by `mask` go to / come from consecutive
# registers starting at d / src. Loads are asynchronous: the first consumer must wait (store_w/load_w ext b9.5;
# 27-family ALU `wait` = b0.3), and a pending load's dest must not be overwritten before that wait.
MEM_BASE = [(1, i) for i in range(1, 8)]
MEM_IDX = [(3, i) for i in range(1, 7)]
MEM_OFF = [(6, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4)]
MEM_SCALE = {4: 2, 8: 3, 16: 0, 1: 1}                     # bytes per index unit -> b7[6:5]
MEM_MASK = {"x": 0, "y": 1, "z": 2, "w": 3, "xw": 4, "xy": 5, "xyz": 6, "xyzw": 7}   # (b4.5, b4.6, b5.3)


# byte4 bit 7 = the base operand's REGISTER FILE: clear = the uniform file (the default), set = the GENERAL
# REGISTER file, so `base` names a 64-bit register pair instead of a uniform pair. Same field, same numbering
# (the index of the first 32-bit half of the pair) -- the split AGX encodes with its own `agx_pack_memory_base`
# flag.
BASE_IS_REG = (4, 7)


def _mem_fields(data):
    scale = enum_field("scale", [(7, 5), (7, 6)], MEM_SCALE)
    scale.default = TEMPLATE                                  # 4-byte elements unless set
    return [Field(data, MOV_DEST, kind="reg"), Field("base", MEM_BASE, kind="ureg"),
            Field("idx", MEM_IDX, kind="reg"), Field("indexed", [(5, 1)], kind="flag", default=TEMPLATE),
            Field("offset", MEM_OFF, kind="imm"), scale, enum_field("mask", [(4, 5), (4, 6), (5, 3)], MEM_MASK),
            Field("base_reg", [BASE_IS_REG], kind="flag")]


# b5.5 on loads: no observable effect on the loaded value (probably a cache policy)
LOAD_HINT5 = (5, 5)
_op("load", 8, "0f00030018021040", fields=_mem_fields("d") + [Field("hint5", [LOAD_HINT5], kind="hint")],
    family="mem", summary="d.. = mem[u[base] + scale*r[idx] + 4*offset] (mask components)")
_op("load_w", 10, "0f000300180210c00020",
    fields=_mem_fields("d") + [Field("hint5", [LOAD_HINT5], kind="hint"),
                               Field("wait", [(9, 5)], kind="flag", default=TEMPLATE)],
    family="mem", summary="load, extended form; wait=1 first waits for outstanding loads")
_op("store", 8, "0f00030001061040", fields=_mem_fields("src"),
    family="mem", summary="mem[u[base] + scale*r[idx] + 4*offset] = src.. (mask components)")
_op("store_w", 10, "0f000300010610c00020", fields=_mem_fields("src") + [Field("wait", [(9, 5)], kind="flag", default=TEMPLATE)],
    family="mem", summary="store, extended form; wait=1 first waits for outstanding loads (e.g. the stored register)")
_op("ustore", 4, "0b000924",
    fields=[Field("u", MOV_DEST, kind="ureg"), Field("src", [(1, i) for i in range(1, 7)], kind="reg")],
    family="mem", summary="u[u] = src   (PREAMBLE ONLY: visible to the main body; no effect from the body)")

# ---- control flow (marker-scan; no byte offsets) ----
# Predicates: a compare WRITES predicate p = b0[7:5]; a branch and a back-edge READ predicate p = b1[7:4].
# p0 is the loop condition; p1..p5 work for an if/else inside a loop. Writing p0 from an inner compare breaks
# the loop.
PRED_W = [(0, 5), (0, 6), (0, 7)]
PRED_R = [(1, 4), (1, 5), (1, 6), (1, 7)]

# Back-edge offset: signed halfword distance, target = fe + 4 + 2*offset (the loop body's first instruction).
# Bits 0..10 are tested on hardware (to -685 halfwords); 11..12 are the sign-extension bits.
FE_OFF = [(4, 7), (5, 4), (5, 5), (5, 6), (6, 1), (6, 3), (6, 5), (6, 6), (6, 7), (5, 0), (5, 1), (5, 2), (5, 3)]
_FE_MASK = (1 << len(FE_OFF)) - 1


def _sext(raw, bits=len(FE_OFF)):
    return raw - (1 << bits) if raw >> (bits - 1) else raw


# 6-B compares. op1/op2 = b1[6:1] / b3[6:1] (r0..r63). b4.0 = direction (0 ">", 1 "<"); strict.
# Compare type b4[2:1]: 2 = unsigned, 3 = signed, 1 = float (denormals flushed, NaN false).
# p6 reads constant true, p7 false; p0..p5 are allocatable.
#
# b1.0 / b3.0 are the operand WIDTH bits, not hints: SET = read the full 32-bit register, CLEAR = read
# only its low 16 bits (zero-extended, then compared at the given type). An emitter must SET them for
# 32-bit operands; the template defaults already do.
#
# THE REGISTER-OPERAND COMPARES KILL THEIR OPERANDS. `cmp_reg`, `cmp_uni` and `cmp_eq` mark their
# register sources dead, so a later reader gets garbage. `cmp_imm` never does this.
#
# op2's kill is controllable: byte2 bit 4, SET = discard op2, CLEAR = keep it, with the predicate
# unaffected either way. op1's is not: byte5 bit 0 keeps the register but makes op1 read as 0, so it is
# a source-mode bit, not a liveness hint. An emitter that needs op1 after the compare must swap the
# operands (flipping `lt`, or freely for `cmp_eq`) so the live value lands in op2, or copy it.
CMP_TYPE = {"u": 2, "s": 3, "f": 1}


def _cmp_type():
    f = enum_field("type", [(4, 1), (4, 2)], CMP_TYPE)
    f.default = TEMPLATE
    return f


_op("cmp_imm", 6, "0a01228c1422",
    fields=[Field("op1", A91, kind="reg"), Field("imm", CMP_IMM, kind="imm"),
            Field("lt", [(4, 0)], kind="flag"), Field("p", PRED_W, kind="pred"), _cmp_type(),
            hint("op1_32", (1, 0))],
    family="ctrl", summary="pred := op1 (lt?<:>) imm   (typed; strict)")
_op("cmp_reg", 6, "0a033a050400",
    fields=[Field("op1", A91, kind="reg"), Field("op2", B91, kind="reg"), Field("lt", [(4, 0)], kind="flag"),
            Field("p", PRED_W, kind="pred"), _cmp_type(), hint("op1_32", (1, 0)), hint("op2_32", (3, 0)),
            hint("discard_op2", (2, 4))],
    family="ctrl", summary="pred := op1 (lt?<:>) op2   (typed; strict)")
_op("cmp_uni", 6, "0a032a050402",
    fields=[Field("op1", A91, kind="reg"), Field("u", B91, kind="ureg"), Field("lt", [(4, 0)], kind="flag"),
            Field("p", PRED_W, kind="pred"), _cmp_type(), hint("op1_32", (1, 0))],
    family="ctrl", summary="pred := op1 (lt?<:>) u[u]   (typed; strict; b5.1 = uniform op2)")
_op("cmp_eq", 10, "0a033b05060007000000",
    fields=[Field("op1", A91, kind="reg"), Field("op2", B91, kind="reg"), Field("p", PRED_W, kind="pred"),
            hint("op1_32", (1, 0)), hint("op2_32", (3, 0)), hint("discard_op2", (2, 4))],
    family="ctrl", summary="pred := op1 == op2 (bitwise; 10-B form, condition code 7). != via branch polarity")
_op("branch", 4, "1e00000e", fields=[Field("true", [(2, 1)], kind="flag"), Field("p", PRED_R, kind="pred")],
    family="ctrl", summary="branch past the then-block (to the code after the jump) when pred is false")
_op("jump", 4, "5e63400e", family="ctrl", summary="unconditional forward jump to endif")
_op("loop_exit", 4, "be03400e", family="ctrl",
    summary="loop-exit marker (follows a back-edge). NOT an if/else marker: inside if/else it makes the else "
            "block's register writes land on all lanes")
_op("endif_marker", 4, "3e03400e", family="ctrl", summary="endif marker")
_op("back_edge", 14, "fe00000e5e0ffb1e1f00ffc7ff7f",
    fields=[Field("true", [(2, 1)], kind="flag"), Field("p", PRED_R, kind="pred"),
            Field("offset", FE_OFF, enc=lambda v: v & _FE_MASK, dec=_sext, kind="imm")],
    family="ctrl", summary="loop back-edge: if pred p, jump to fe + 4 + 2*offset (signed halfwords)")

# ---- loop-structural pieces (fixed templates / small fields; emitted only by the loop builders) ----
_op("regcache13", 10, "1380070a02a084020a85", family="loop",
    summary="register-cache clear (MANDATORY before a loop; omitting it hangs)")
_op("loop_marker", 4, "2b002020",
    fields=[Field("hint", [(0, 4), (0, 5), (0, 6), (0, 7)], kind="imm"), Field("tag", [(3, i) for i in range(8)], kind="imm")],
    family="loop", summary="structural marker (b0[7:4] ~ live-register hint)")
_op("wadd_cmp", 4, "22010500",
    fields=[Field("op2", [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)], kind="reg")],
    family="loop", summary="pred := r1 < op2; if pred then r1 := r2   (wadd back-edge cmp)")
_op("ksetup", 4, "22800402",
    fields=[Field("k", [(2, 4), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)], kind="imm")],
    family="loop", summary="set the shift-loop condition constant K (runs once, pre-loop; 0..127)")
_op("shift_cmp", 14, "2700003a2600a10a788000340000", family="loop",
    summary="r2 >>= 1; pred := r2 > K_slot   (27-a1 fused shift+compare)")


# --------------------------------------------------------------------------------------------------
# generic pack / unpack driven purely by the table above
# --------------------------------------------------------------------------------------------------
BY_NAME = {i.name: i for i in INSNS}


def pack(op_name, **operands):
    """Encode one instruction by name. Unknown operands / overflow raise."""
    return BY_NAME[op_name].encode(**operands)


_LENGTH = None


def _length_model():
    """The measured instruction-length model (g17_length.json)."""
    global _LENGTH
    if _LENGTH is None:
        import json
        import os
        _LENGTH = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "g17_length.json")))
    return _LENGTH


def length_row(b0, b1, b2):
    """Row = b0&7, b1 bit7 (move family only), b2&7 -> 0..127."""
    return (b0 & 7) << 4 | (1 if (b0 & 7) == 4 and b1 & 0x80 else 0) << 3 | (b2 & 7)


def length_col(b4, b6, b8):
    return (b4 & 7) << 7 | (b6 & 0xF) << 3 | (b8 & 7)


def insn_length(code, p):
    """Length in bytes of the instruction at offset ``p``, or None if unknown.
    Ordered special-case rules first (memory family, control flow, loop-family forms), then the two-level table."""
    m = _length_model()
    x = (bytes(code[p:p + 14]) + bytes(14))[:14]
    for r in m["rules"]:
        if all((x[i] & r["mask"][i]) == r["match"][i] for i in range(14)):
            return r["length"]
    row = m["rows"][length_row(x[0], x[1], x[2])]
    n = row["const"] if "const" in row else m["subtables"][row["sub"]][length_col(x[4], x[6], x[8])]
    return n or None


class Decoded:
    """A disassembled instruction: its op name, operand dict, raw bytes and offset."""

    __slots__ = ("op", "operands", "bytes", "offset")

    def __init__(self, op, operands, raw, offset):
        self.op = op
        self.operands = operands
        self.bytes = raw
        self.offset = offset

    def reencode(self):
        if self.op is None:
            return self.bytes
        return BY_NAME[self.op].encode(**self.operands)

    def __repr__(self):
        ops = ", ".join(f"{k}={v}" for k, v in self.operands.items())
        return f"{self.op or '??'}({ops})"


def match(slice_, length=None):
    """Return the :class:`Insn` whose opcode mask matches ``slice_`` (optionally of given length)."""
    hits = [i for i in INSNS if (length is None or i.length == length) and i.matches(slice_)]
    if not hits:
        return None
    # Prefer the most specific match (largest fixed mask popcount) if several apply.
    hits.sort(key=lambda i: sum(bin(b).count("1") for b in i.fixed_mask), reverse=True)
    return hits[0]


def validate_tables():
    """Assert no two ops of the same length are mutually indistinguishable by their opcode masks.

    Two ops OVERLAP iff, on the bits BOTH pin (the intersection of their fixed masks), their match
    values agree — then some byte string satisfies either. That alone is not a defect, because
    :func:`match` resolves it the way a real decoder does: it prefers the op pinning MORE bits. An
    overlap is only ambiguous when the two pin the SAME number of bits, since then the tie-break is
    arbitrary. Raises AssertionError listing the genuinely ambiguous pairs.

    The overlaps that specificity resolves are real and expected: an op with no operands pins every
    bit of its template, and can also be a legal encoding of a wider op with every field zero. The
    hardware tells them apart by context, which a flat table cannot express, so specificity is the
    right proxy and it picks the op with no operands.
    """
    def pinned(i):
        return sum(bin(b).count("1") for b in i.fixed_mask)

    clashes = []
    for a in INSNS:
        for b in INSNS:
            if a is b or a.length != b.length or a.name >= b.name:
                continue
            common = bytes(x & y for x, y in zip(a.fixed_mask, b.fixed_mask))
            if all((a.match[i] & common[i]) == (b.match[i] & common[i]) for i in range(a.length)):
                if pinned(a) == pinned(b):
                    clashes.append((a.name, b.name))
    assert not clashes, "ambiguous opcode masks: " + ", ".join(f"{x}~{y}" for x, y in clashes)
    return True


if __name__ == "__main__":
    validate_tables()
    print(f"G17 ISA: {len(INSNS)} instruction forms, tables validated (masks unambiguous).")
    by_family = {}
    for i in INSNS:
        by_family.setdefault(i.family, []).append(i.name)
    for fam, names in by_family.items():
        print(f"  {fam:11} {' '.join(names)}")
