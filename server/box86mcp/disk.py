"""Hard disk images: building bootable DOS disks and copying files onto them.

Images are raw (86Box .img). Only the first primary partition is handled.
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from pathlib import Path

from .vm import FIXED_MTIME, MTOOLS_ENV, VOLUME_SERIAL, _tool

SECTOR = 512

# 512 MB CompactFlash (e.g. SanDisk SDCFB-512): 993 cylinders, 16 heads, 63 sectors.
CF_512MB = (993, 16, 63)


def partition_offset(img: str | os.PathLike) -> int:
    """Byte offset of the first non-empty primary partition."""
    with open(img, "rb") as f:
        mbr = f.read(SECTOR)
    if mbr[510:512] != b"\x55\xaa":
        raise ValueError(f"{img}: no MBR signature")
    for i in range(4):
        entry = mbr[446 + 16 * i: 462 + 16 * i]
        ptype = entry[4]
        start, = struct.unpack("<I", entry[8:12])
        if ptype and start:
            return start * SECTOR
    raise ValueError(f"{img}: no partition")


def _chs(lba: int, heads: int, spt: int) -> bytes:
    c, rem = divmod(lba, heads * spt)
    h, s = divmod(rem, spt)
    c = min(c, 1023)
    return bytes([h, ((s + 1) & 0x3F) | ((c >> 2) & 0xC0), c & 0xFF])


def make_dos_disk(dest: str | os.PathLike, base_img: str | os.PathLike,
                  geometry: tuple[int, int, int] = CF_512MB) -> Path:
    """Create a bootable FAT16 DOS disk of the given CHS geometry.

    The MBR code, DOS boot sector code, IO.SYS, MSDOS.SYS and COMMAND.COM come
    from base_img (an existing bootable DOS disk), which is the equivalent of
    FDISK + FORMAT /S without needing the DOS install media.
    """
    cyls, heads, spt = geometry
    total = cyls * heads * spt
    start = spt  # first partition starts on head 1, as DOS FDISK does
    size = total - start
    dest = Path(dest)
    base_off = partition_offset(base_img)

    with open(base_img, "rb") as f:
        mbr_code = f.read(446)
        f.seek(base_off)
        vbr = f.read(SECTOR)

    entry = (bytes([0x80]) + _chs(start, heads, spt) + bytes([0x06]) + _chs(total - 1, heads, spt)
             + struct.pack("<II", start, size))
    with open(dest, "wb") as f:
        f.write(mbr_code + entry + bytes(48) + b"\x55\xaa")
        f.truncate(total * SECTOR)

    with tempfile.TemporaryDirectory() as tmp:
        vbr_file = Path(tmp) / "vbr.bin"
        vbr_file.write_bytes(vbr)
        part = f"{dest}@@{start * SECTOR}"
        subprocess.run([_tool("mformat"), "-i", part, "-B", str(vbr_file), "-T", str(size), "-h", str(heads),
                        "-s", str(spt), "-H", str(start), "-r", "32", "-N", VOLUME_SERIAL, "::"],
                       check=True, capture_output=True, env=MTOOLS_ENV)

        # IO.SYS and MSDOS.SYS must be the first two root entries and IO.SYS must
        # start at the first data cluster, so copy them first, onto an empty disk.
        base = f"{base_img}@@{base_off}"
        for name in ("IO.SYS", "MSDOS.SYS", "COMMAND.COM"):
            local = Path(tmp) / name
            # -m both ways: the files keep the base image's dates rather than today's. The target is
            # a bare name in cwd: mtools takes a Windows path's "F:" for one of its own drives.
            subprocess.run([_tool("mcopy"), "-m", "-i", base, f"::{name}", name], check=True,
                           capture_output=True, env=MTOOLS_ENV, cwd=tmp)
            subprocess.run([_tool("mcopy"), "-m", "-i", part, str(local), f"::{name}"], check=True,
                           capture_output=True, env=MTOOLS_ENV)
        subprocess.run([_tool("mattrib"), "-i", part, "+h", "+s", "+r", "::IO.SYS", "::MSDOS.SYS"],
                       check=True, capture_output=True, env=MTOOLS_ENV)
    return dest


def put_files(img: str | os.PathLike, files: list[str | os.PathLike], dest_dir: str = "\\") -> list[str]:
    """Copy host files/directories into a directory of the image's first partition.

    The emulator must not have the image open: 86Box caches sectors, so writes
    from outside while it runs would corrupt the file system.
    """
    part = f"{img}@@{partition_offset(img)}"
    target = "::" + dest_dir.replace("\\", "/").strip("/")
    if target != "::":
        # mmd fails if it exists; that is fine.
        subprocess.run([_tool("mmd"), "-D", "s", "-i", part, target], capture_output=True, env=MTOOLS_ENV)
    copied = []
    for src in files:
        src = Path(src).expanduser()
        r = subprocess.run([_tool("mcopy"), "-i", part, "-s", "-o", "-m", str(src), target + "/"],
                           capture_output=True, text=True, env=MTOOLS_ENV)
        if r.returncode:
            raise RuntimeError(f"mcopy {src} failed: {r.stderr.strip()}")
        copied.append(src.name.upper())
    return copied


def write_text(img: str | os.PathLike, dos_path: str, text: str) -> None:
    """Write a text file (CRLF line endings) into the image's first partition."""
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / Path(dos_path.replace("\\", "/")).name
        local.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("ascii"))
        os.utime(local, (FIXED_MTIME, FIXED_MTIME))
        part = f"{img}@@{partition_offset(img)}"
        subprocess.run([_tool("mcopy"), "-i", part, "-o", "-m", str(local), "::" + dos_path.replace("\\", "/")],
                       check=True, capture_output=True, env=MTOOLS_ENV)

