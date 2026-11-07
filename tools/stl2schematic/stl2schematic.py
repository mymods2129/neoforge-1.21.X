#!/usr/bin/env python3
"""
stl2schematic - convert .stl meshes into Create mod schematics.

Create's Schematic Table / Schematicannon read the vanilla *structure* format:
a gzip-compressed NBT file with `size`, `palette`, `blocks` and `entities`.
Drop the resulting .nbt into `.minecraft/schematics/` and it shows up in
the Schematic Table.

Pipeline:
    1. Load the STL (binary or ASCII).
    2. Rotate from CAD's Z-up into Minecraft's Y-up and scale to blocks.
    3. Voxelise the surface by densely sampling every triangle.
    4. Optionally fill the inside (flood-fill the outside, everything
       that isn't reached is interior).
    5. Write the structure NBT.

Only dependency: numpy.
"""
from __future__ import annotations

import argparse
import gzip
import io
import math
import struct
import sys
from pathlib import Path

import numpy as np

# Minecraft data versions. Create reads older versions fine; using the one
# that matches your game avoids DataFixer work when loading.
DATA_VERSIONS = {
    "1.20.1": 3465,
    "1.21": 3953,
    "1.21.1": 3955,
}

# --------------------------------------------------------------------------
# STL loading
# --------------------------------------------------------------------------


def load_stl(path: Path) -> np.ndarray:
    """Return an (N, 3, 3) float64 array of triangles."""
    data = path.read_bytes()
    if len(data) >= 84:
        (count,) = struct.unpack_from("<I", data, 80)
        # Binary STLs have an exact size. Some exporters write "solid" in the
        # binary header, so the size check is more reliable than the prefix.
        if 84 + count * 50 == len(data):
            return _load_binary_stl(data, count)
    if data.lstrip()[:5].lower() == b"solid":
        return _load_ascii_stl(data)
    raise ValueError(f"{path} does not look like a valid STL file")


def _load_binary_stl(data: bytes, count: int) -> np.ndarray:
    dtype = np.dtype(
        [("normal", "<f4", 3), ("verts", "<f4", (3, 3)), ("attr", "<u2")]
    )
    records = np.frombuffer(data, dtype=dtype, count=count, offset=84)
    return records["verts"].astype(np.float64)


def _load_ascii_stl(data: bytes) -> np.ndarray:
    verts = []
    for line in data.decode("ascii", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            verts.append([float(v) for v in parts[1:]])
    if not verts or len(verts) % 3:
        raise ValueError("ASCII STL has no triangles or a broken vertex count")
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------


def to_minecraft_axes(tris: np.ndarray, up: str) -> np.ndarray:
    """Rotate so the model's `up` axis becomes Minecraft +Y.

    These are proper rotations (no mirroring), so left and right wings stay
    on the correct side.
    """
    x, y, z = tris[..., 0], tris[..., 1], tris[..., 2]
    if up == "z":  # most CAD tools (Fusion 360, SolidWorks, Onshape, FreeCAD)
        out = (x, z, -y)
    elif up == "y":  # Blender exports with "Y up", many game tools
        out = (x, y, z)
    elif up == "x":
        out = (-y, x, z)
    else:
        raise ValueError(f"unknown up axis {up!r}")
    return np.stack(out, axis=-1)


def rotate_yaw(tris: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate around Minecraft's Y axis in 90 degree steps."""
    steps = (degrees // 90) % 4
    x, y, z = tris[..., 0], tris[..., 1], tris[..., 2]
    for _ in range(steps):
        x, z = -z, x
    return np.stack((x, y, z), axis=-1)


def scale_to_blocks(
    tris: np.ndarray, scale: float | None, size: float | None
) -> tuple[np.ndarray, float]:
    """Scale/translate so the model's min corner sits at the origin.

    `scale` = blocks per model unit. `size` = target length (in blocks) of
    the longest side. Exactly one should be given.
    """
    lo = tris.reshape(-1, 3).min(axis=0)
    hi = tris.reshape(-1, 3).max(axis=0)
    extent = hi - lo
    if size is not None:
        longest = float(extent.max())
        if longest <= 0:
            raise ValueError("model has zero size")
        scale = size / longest
    assert scale is not None
    return (tris - lo) * scale, scale


# --------------------------------------------------------------------------
# Voxelisation
# --------------------------------------------------------------------------


def voxelize_surface(tris: np.ndarray, step: float = 0.3) -> np.ndarray:
    """Mark every voxel touched by a triangle.

    Each triangle is sampled on a barycentric grid whose spacing is at most
    `step` blocks, which yields a 26-connected shell. A 26-connected shell
    is a closed wall for a 6-connected flood fill, which is what fill_interior
    relies on.
    """
    pts = tris.reshape(-1, 3)
    # A side exactly N blocks long gives N blocks; points on the far face
    # are clipped into the last layer.
    dims = np.maximum(np.ceil(pts.max(axis=0) - 1e-9).astype(int), 1)
    grid = np.zeros(dims, dtype=bool)

    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    longest_edge = np.max(
        np.stack(
            [
                np.linalg.norm(b - a, axis=1),
                np.linalg.norm(c - b, axis=1),
                np.linalg.norm(a - c, axis=1),
            ]
        ),
        axis=0,
    )
    subdiv = np.maximum(np.ceil(longest_edge / step).astype(int), 1)

    def mark(points: np.ndarray) -> None:
        idx = np.floor(points).astype(int)
        np.clip(idx, 0, dims - 1, out=idx)
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    # Always mark the vertices themselves.
    mark(pts)

    # Group triangles by subdivision count so each group is one vectorised op.
    for n in np.unique(subdiv):
        sel = subdiv == n
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = (i + j) <= n
        u = (i[keep] / n)[:, None]  # (K, 1)
        v = (j[keep] / n)[:, None]
        ta, tb, tc = a[sel], b[sel], c[sel]  # (M, 3)
        # Limit memory: process large groups in chunks.
        k = u.shape[0]
        chunk = max(1, 4_000_000 // max(k, 1))
        for s in range(0, ta.shape[0], chunk):
            A, B, C = ta[s : s + chunk], tb[s : s + chunk], tc[s : s + chunk]
            points = (
                A[:, None, :]
                + u[None] * (B - A)[:, None, :]
                + v[None] * (C - A)[:, None, :]
            )
            mark(points.reshape(-1, 3))
    return grid


def fill_interior(surface: np.ndarray) -> np.ndarray:
    """Return a mask of voxels enclosed by the surface.

    Flood-fills "outside" from the border of a padded grid with 6-connectivity,
    using axis sweeps (fast in numpy) repeated until nothing changes. Anything
    not outside and not surface is interior.
    """
    padded = np.pad(surface, 1, constant_values=False)
    outside = np.zeros_like(padded)
    outside[0, :, :] = outside[-1, :, :] = True
    outside[:, 0, :] = outside[:, -1, :] = True
    outside[:, :, 0] = outside[:, :, -1] = True
    outside &= ~padded

    while True:
        before = int(outside.sum())
        for axis in range(3):
            for flip in (False, True):
                outside = _sweep(outside, padded, axis, flip)
        if int(outside.sum()) == before:
            break

    interior = ~outside & ~padded
    return interior[1:-1, 1:-1, 1:-1]


def _sweep(outside: np.ndarray, wall: np.ndarray, axis: int, flip: bool) -> np.ndarray:
    """Propagate `outside` along one axis direction until a wall is hit."""
    if flip:
        outside = np.flip(outside, axis)
        wall = np.flip(wall, axis)
    n = outside.shape[axis]
    shape = [1, 1, 1]
    shape[axis] = n
    idx = np.arange(n).reshape(shape)
    last_wall = np.maximum.accumulate(np.where(wall, idx, -1), axis=axis)
    last_out = np.maximum.accumulate(np.where(outside, idx, -1), axis=axis)
    result = outside | ((last_out > last_wall) & ~wall)
    if flip:
        result = np.flip(result, axis)
    return np.ascontiguousarray(result)


# --------------------------------------------------------------------------
# NBT writing (just the subset needed for structure files)
# --------------------------------------------------------------------------

TAG_END, TAG_INT, TAG_STRING, TAG_LIST, TAG_COMPOUND = 0, 3, 8, 9, 10


class NBTWriter:
    def __init__(self) -> None:
        self.buf = io.BytesIO()

    def _w(self, fmt: str, *vals) -> None:
        self.buf.write(struct.pack(">" + fmt, *vals))

    def _str(self, s: str) -> None:
        raw = s.encode("utf-8")
        self._w("H", len(raw))
        self.buf.write(raw)

    def _payload(self, tag: int, value) -> None:
        if tag == TAG_INT:
            self._w("i", value)
        elif tag == TAG_STRING:
            self._str(value)
        elif tag == TAG_LIST:
            elem_tag, items = value
            self._w("b", elem_tag if items else TAG_END)
            self._w("i", len(items))
            for item in items:
                self._payload(elem_tag, item)
        elif tag == TAG_COMPOUND:
            for name, (t, v) in value.items():
                self._w("b", t)
                self._str(name)
                self._payload(t, v)
            self._w("b", TAG_END)
        else:
            raise ValueError(tag)

    def root(self, compound: dict) -> bytes:
        self._w("b", TAG_COMPOUND)
        self._str("")
        self._payload(TAG_COMPOUND, compound)
        return self.buf.getvalue()


def parse_block_state(spec: str) -> dict:
    """'minecraft:oak_stairs[facing=north,half=top]' -> palette compound."""
    name, props = spec, {}
    if "[" in spec:
        name, rest = spec.split("[", 1)
        for kv in rest.rstrip("]").split(","):
            if kv.strip():
                k, v = kv.split("=", 1)
                props[k.strip()] = v.strip()
    if ":" not in name:
        name = "minecraft:" + name
    entry = {"Name": (TAG_STRING, name)}
    if props:
        entry["Properties"] = (
            TAG_COMPOUND,
            {k: (TAG_STRING, v) for k, v in props.items()},
        )
    return entry


def write_structure(
    path: Path, layers: list[tuple[str, np.ndarray]], data_version: int
) -> int:
    """Write a structure .nbt. `layers` is [(block_spec, bool mask), ...];
    later layers win where they overlap. Returns the block count."""
    dims = layers[0][1].shape
    state = np.full(dims, -1, dtype=np.int32)
    palette_specs: list[str] = []
    for spec, mask in layers:
        if spec not in palette_specs:
            palette_specs.append(spec)
        state[mask] = palette_specs.index(spec)

    xs, ys, zs = np.nonzero(state >= 0)
    states = state[xs, ys, zs]
    blocks = [
        {
            "pos": (TAG_LIST, (TAG_INT, [int(x), int(y), int(z)])),
            "state": (TAG_INT, int(s)),
        }
        for x, y, z, s in zip(xs, ys, zs, states)
    ]
    root = {
        "DataVersion": (TAG_INT, data_version),
        "size": (TAG_LIST, (TAG_INT, [int(d) for d in dims])),
        "palette": (TAG_LIST, (TAG_COMPOUND, [parse_block_state(p) for p in palette_specs])),
        "blocks": (TAG_LIST, (TAG_COMPOUND, blocks)),
        "entities": (TAG_LIST, (TAG_COMPOUND, [])),
    }
    raw = NBTWriter().root(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(raw)
    return len(blocks)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert an STL mesh into a Create mod schematic (.nbt).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path, help="input .stl file")
    p.add_argument(
        "-o", "--output", type=Path,
        help="output .nbt (default: input name with .nbt next to the STL)",
    )
    size = p.add_mutually_exclusive_group()
    size.add_argument(
        "--size", type=float, default=None,
        help="length of the model's longest side in blocks (default 64)",
    )
    size.add_argument(
        "--scale", type=float, default=None,
        help="blocks per model unit, e.g. 0.1 = 1 block per 10 mm",
    )
    p.add_argument("--up", choices=["x", "y", "z"], default="z",
                   help="which axis is 'up' in your CAD model")
    p.add_argument("--yaw", type=int, default=0, choices=[0, 90, 180, 270],
                   help="rotate the model around the vertical axis")
    p.add_argument("--block", default="minecraft:white_concrete",
                   help="block for the outer shell (block state syntax allowed)")
    p.add_argument("--fill-block", default=None,
                   help="block for the interior (default: same as --block)")
    p.add_argument("--hollow", action="store_true",
                   help="only place the outer shell, leave the inside empty")
    p.add_argument("--mc-version", choices=sorted(DATA_VERSIONS), default="1.21.1",
                   help="Minecraft version to tag the schematic with")
    return p


def ask_with_dialogs() -> list[str] | None:
    """Started with no arguments (e.g. the Run button or a double-click):
    ask for the file and main options with simple dialogs instead."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        while True:
            stl = filedialog.askopenfilename(
                title="Choose an STL file to convert",
                filetypes=[("STL files", "*.stl")],
            )
            if not stl:
                return None
            try:
                load_stl(Path(stl))
                break
            except (ValueError, OSError) as e:
                messagebox.showerror(
                    "Not an STL file",
                    f"{Path(stl).name} isn't an STL file this tool can read.\n\n"
                    f"Export your model from your CAD program as .stl and "
                    f"pick that file.\n\n({e})",
                    parent=root,
                )
        size = simpledialog.askinteger(
            "Size", "Length of the longest side, in blocks:",
            initialvalue=64, minvalue=1, maxvalue=2000, parent=root,
        )
        if size is None:
            return None
        hollow = messagebox.askyesno(
            "Hollow?",
            "Build only the outer shell (hollow)?\n\n"
            "Yes = fewer blocks, smaller file (recommended for planes)\n"
            "No = completely solid",
            parent=root,
        )
        block = simpledialog.askstring(
            "Block", "Block to build with:",
            initialvalue="minecraft:white_concrete", parent=root,
        )
        if not block:
            return None
        out = filedialog.asksaveasfilename(
            title="Save schematic as",
            initialfile=Path(stl).with_suffix(".nbt").name,
            initialdir=str(Path(stl).parent),
            defaultextension=".nbt",
            filetypes=[("Schematic", "*.nbt")],
        )
        if not out:
            return None
    finally:
        root.destroy()

    argv = [stl, "-o", out, "--size", str(size), "--block", block]
    if hollow:
        argv.append("--hollow")
    return argv


def main(argv: list[str] | None = None) -> int:
    if argv is None and len(sys.argv) == 1:
        argv = ask_with_dialogs()
        if argv is None:
            print("Cancelled.")
            return 0
    args = build_parser().parse_args(argv)
    if args.size is None and args.scale is None:
        args.size = 64

    try:
        tris = load_stl(args.input)
    except (ValueError, OSError) as e:
        print(f"Error: couldn't read {args.input}: {e}", file=sys.stderr)
        print("Pick a .stl file exported from your CAD program.", file=sys.stderr)
        return 1
    print(f"Loaded {len(tris):,} triangles from {args.input}")

    tris = to_minecraft_axes(tris, args.up)
    tris = rotate_yaw(tris, args.yaw)
    tris, scale = scale_to_blocks(tris, args.scale, args.size)

    surface = voxelize_surface(tris)
    dims = surface.shape
    print(f"Grid: {dims[0]} x {dims[1]} x {dims[2]} blocks "
          f"(X x Y x Z), {scale:.4g} blocks per model unit")

    layers = [(args.block, surface)]
    if not args.hollow:
        interior = fill_interior(surface)
        n_in = int(interior.sum())
        if n_in == 0:
            print("Note: no enclosed interior found. The mesh may have holes "
                  "or be thinner than a block; output is shell only.")
        layers.append((args.fill_block or args.block, interior))

    out = args.output or args.input.with_suffix(".nbt")
    count = write_structure(out, layers, DATA_VERSIONS[args.mc_version])
    kb = out.stat().st_size / 1024
    print(f"Wrote {count:,} blocks to {out} ({kb:.1f} KB)")
    if kb > 256:
        print("Warning: Create's default upload limit is 256 KB "
              "(maxTotalSchematicSize in the server config). Raise it, use "
              "--hollow, or use a smaller --size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
