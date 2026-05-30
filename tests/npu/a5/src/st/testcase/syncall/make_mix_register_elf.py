#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
#
# Build a compact registration ELF for runtime mix-kernel handle launch.
#
# Bisheng fat objects contain host launch stubs plus nested device ELFs.  The
# runtime's rtRegisterAllKernel parser uses the outer ELF function symbol values
# as device-code offsets, so registering the fat object launches host .text.
# This tool extracts the nested AIC/AIV device ELFs and writes a small ELF whose
# .text is the concatenated device code and whose symbols point at that code.

from __future__ import annotations

import argparse
import logging
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ELF_MAGIC = b"\x7fELF"
ET_REL = 1
EV_CURRENT = 1
SHT_NULL = 0
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_NOTE = 7
SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4
STB_LOCAL = 0
STB_GLOBAL = 1
STT_SECTION = 3
STT_FUNC = 2

EHDR = struct.Struct("<16sHHIQQQIHHHHHH")
SHDR = struct.Struct("<IIQQQQIIQQ")
SYMENT = struct.Struct("<IBBHQQ")


def align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return (value + alignment - 1) // alignment * alignment


def st_info(bind: int, typ: int) -> int:
    return (bind << 4) | (typ & 0xF)


@dataclass
class Section:
    name: str
    typ: int
    flags: int
    align: int
    data: bytes
    link: int = 0
    info: int = 0
    entsize: int = 0
    offset: int = 0
    name_offset: int = 0


@dataclass
class Symbol:
    name: str
    info: int
    other: int
    shndx: int
    value: int
    size: int


@dataclass
class ParsedElf:
    machine: int
    flags: int
    sections: dict[str, bytes]
    symbols: list[Symbol]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_elf(data: bytes, label: str) -> ParsedElf:
    require(data.startswith(ELF_MAGIC), f"{label}: not an ELF payload")
    require(len(data) >= EHDR.size, f"{label}: truncated ELF header")
    (
        ident,
        _etype,
        machine,
        _version,
        _entry,
        _phoff,
        shoff,
        flags,
        _ehsize,
        _phentsize,
        _phnum,
        shentsize,
        shnum,
        shstrndx,
    ) = EHDR.unpack_from(data, 0)
    require(ident[4] == 2 and ident[5] == 1, f"{label}: expected ELF64 little-endian")
    require(shentsize == SHDR.size, f"{label}: unexpected section header size {shentsize}")
    require(shoff + shnum * shentsize <= len(data), f"{label}: section table is truncated")
    raw_sections = [SHDR.unpack_from(data, shoff + idx * shentsize) for idx in range(shnum)]
    require(shstrndx < shnum, f"{label}: invalid shstrndx {shstrndx}")
    shstr = raw_section_data(data, raw_sections[shstrndx])

    section_names: list[str] = []
    sections: dict[str, bytes] = {}
    for raw in raw_sections:
        name = read_c_string(shstr, raw[0])
        section_names.append(name)
        if name:
            sections[name] = raw_section_data(data, raw)

    symbols: list[Symbol] = []
    for raw in raw_sections:
        if raw[1] != SHT_SYMTAB:
            continue
        strtab_idx = raw[6]
        require(strtab_idx < shnum, f"{label}: invalid symtab link {strtab_idx}")
        strtab = raw_section_data(data, raw_sections[strtab_idx])
        symtab = raw_section_data(data, raw)
        require(raw[9] == SYMENT.size, f"{label}: unexpected symbol size {raw[9]}")
        for off in range(0, len(symtab), SYMENT.size):
            st_name, info, other, shndx, value, size = SYMENT.unpack_from(symtab, off)
            symbols.append(Symbol(read_c_string(strtab, st_name), info, other, shndx, value, size))

    return ParsedElf(machine=machine, flags=flags, sections=sections, symbols=symbols)


def raw_section_data(data: bytes, raw: tuple[int, int, int, int, int, int, int, int, int, int]) -> bytes:
    _name, typ, _flags, _addr, offset, size, _link, _info, _align, _entsize = raw
    if typ == SHT_NULL or size == 0:
        return b""
    require(offset + size <= len(data), "section contents exceed ELF size")
    return data[offset:offset + size]


def read_c_string(data: bytes, offset: int) -> str:
    if offset >= len(data):
        return ""
    end = data.find(b"\0", offset)
    if end < 0:
        end = len(data)
    return data[offset:end].decode("utf-8", errors="replace")


def extract_nested_rel_elf(path: Path) -> ParsedElf:
    outer = parse_elf(path.read_bytes(), str(path))
    rel = outer.sections.get("__aicore_rel_binary")
    require(rel is not None, f"{path}: missing __aicore_rel_binary")
    start = rel.find(ELF_MAGIC)
    require(start >= 0, f"{path}: __aicore_rel_binary does not contain an ELF")
    return parse_elf(rel[start:], f"{path}:__aicore_rel_binary")


def kernel_functions(elf: ParsedElf, suffix: str) -> list[Symbol]:
    funcs = []
    for sym in elf.symbols:
        bind = sym.info >> 4
        typ = sym.info & 0xF
        if bind == STB_GLOBAL and typ == STT_FUNC and sym.name.endswith(suffix):
            funcs.append(sym)
    require(funcs, f"no global function symbols ending with {suffix}")
    return funcs


def make_strtab(names: list[str]) -> tuple[bytes, dict[str, int]]:
    data = bytearray(b"\0")
    offsets: dict[str, int] = {"": 0}
    for name in names:
        if name in offsets:
            continue
        offsets[name] = len(data)
        data.extend(name.encode("utf-8"))
        data.append(0)
    return bytes(data), offsets


def build_output(aic: ParsedElf, aiv: ParsedElf) -> bytes:
    aic_text = aic.sections.get(".text")
    aiv_text = aiv.sections.get(".text")
    require(aic_text is not None and aiv_text is not None, "nested device ELF missing .text")
    aiv_base = align_up(len(aic_text), 4)
    text = aic_text + (b"\0" * (aiv_base - len(aic_text))) + aiv_text

    funcs: list[Symbol] = []
    funcs.extend(
        Symbol(sym.name, sym.info, sym.other, 1, sym.value, sym.size) for sym in kernel_functions(aic, "_mix_aic")
    )
    funcs.extend(
        Symbol(sym.name, sym.info, sym.other, 1, aiv_base + sym.value, sym.size)
        for sym in kernel_functions(aiv, "_mix_aiv")
    )

    sections: list[Section] = [
        Section("", SHT_NULL, 0, 0, b""),
        Section(".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 4, text),
    ]

    stack = aic.sections.get(".ascend.stack.size.record")
    if stack:
        sections.append(Section(".ascend.stack.size.record", SHT_PROGBITS, 0, 1, stack))

    for source in (aic, aiv):
        for name in sorted(source.sections):
            if name.startswith(".ascend.meta."):
                sections.append(Section(name, SHT_NOTE, 0, 4, source.sections[name]))

    local_symbols = [
        Symbol("", 0, 0, 0, 0, 0),
        Symbol("", st_info(STB_LOCAL, STT_SECTION), 0, 1, 0, 0),
    ]
    all_symbols = local_symbols + funcs
    strtab, sym_name_offsets = make_strtab([sym.name for sym in all_symbols])
    symtab = bytearray()
    for sym in all_symbols:
        symtab.extend(SYMENT.pack(sym_name_offsets[sym.name], sym.info, sym.other, sym.shndx, sym.value, sym.size))

    symtab_index = len(sections)
    strtab_index = symtab_index + 1
    shstrtab_index = symtab_index + 2
    sections.append(Section(".symtab", SHT_SYMTAB, 0, 8, bytes(symtab), link=strtab_index,
                            info=len(local_symbols), entsize=SYMENT.size))
    sections.append(Section(".strtab", SHT_STRTAB, 0, 1, strtab))

    shstrtab, section_name_offsets = make_strtab([section.name for section in sections] + [".shstrtab"])
    sections.append(Section(".shstrtab", SHT_STRTAB, 0, 1, shstrtab))
    for section in sections:
        section.name_offset = section_name_offsets[section.name]

    offset = EHDR.size
    body = bytearray(b"\0" * EHDR.size)
    for section in sections[1:]:
        offset = align_up(len(body), section.align)
        body.extend(b"\0" * (offset - len(body)))
        section.offset = offset
        body.extend(section.data)

    shoff = align_up(len(body), 8)
    body.extend(b"\0" * (shoff - len(body)))
    for section in sections:
        body.extend(
            SHDR.pack(
                section.name_offset,
                section.typ,
                section.flags,
                0,
                section.offset,
                len(section.data),
                section.link,
                section.info,
                section.align,
                section.entsize,
            )
        )

    ident = bytearray(16)
    ident[:4] = ELF_MAGIC
    ident[4] = 2
    ident[5] = 1
    ident[6] = EV_CURRENT
    header = EHDR.pack(
        bytes(ident),
        ET_REL,
        aic.machine,
        EV_CURRENT,
        0,
        0,
        shoff,
        aic.flags,
        EHDR.size,
        0,
        0,
        SHDR.size,
        len(sections),
        shstrtab_index,
    )
    body[:EHDR.size] = header
    return bytes(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("aic_object", type=Path)
    parser.add_argument("aiv_object", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    try:
        aic = extract_nested_rel_elf(args.aic_object)
        aiv = extract_nested_rel_elf(args.aiv_object)
        require(aic.machine == aiv.machine, "AIC/AIV device ELFs use different machines")
        args.output.write_bytes(build_output(aic, aiv))
    except Exception as exc:  # noqa: BLE001 - keep build error concise.
        logger.error("make_mix_register_elf.py: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
