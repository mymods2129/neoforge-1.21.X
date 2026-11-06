"""Tests for stl2schematic. Run with:  python -m unittest test_stl2schematic"""
import gzip
import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

import stl2schematic as s2s


# ---- tiny independent NBT reader, used to verify the writer ---------------

def read_nbt(data: bytes):
    pos = 0

    def take(fmt):
        nonlocal pos
        vals = struct.unpack_from(">" + fmt, data, pos)
        pos += struct.calcsize(">" + fmt)
        return vals[0]

    def string():
        nonlocal pos
        n = take("H")
        s = data[pos:pos + n].decode()
        pos += n
        return s

    def payload(tag):
        if tag == 1: return take("b")
        if tag == 2: return take("h")
        if tag == 3: return take("i")
        if tag == 4: return take("q")
        if tag == 5: return take("f")
        if tag == 6: return take("d")
        if tag == 8: return string()
        if tag == 9:
            et, n = take("b"), take("i")
            return [payload(et) for _ in range(n)]
        if tag == 10:
            out = {}
            while True:
                t = take("b")
                if t == 0:
                    return out
                name = string()
                out[name] = payload(t)
        raise ValueError(tag)

    assert take("b") == 10
    string()
    return payload(10)


# ---- mesh helpers ----------------------------------------------------------

def cube_tris(size=10.0):
    v = np.array([[x, y, z] for x in (0, size) for y in (0, size) for z in (0, size)])
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, c, d in faces:
        tris += [[v[a], v[b], v[c]], [v[a], v[c], v[d]]]
    return np.array(tris, dtype=float)


def sphere_tris(r=10.0, n=24):
    tris = []
    for i in range(n):
        t0, t1 = math.pi * i / n, math.pi * (i + 1) / n
        for j in range(2 * n):
            p0, p1 = 2 * math.pi * j / (2 * n), 2 * math.pi * (j + 1) / (2 * n)
            pt = lambda t, p: [r * math.sin(t) * math.cos(p), r * math.sin(t) * math.sin(p), r * math.cos(t)]
            a, b, c, d = pt(t0, p0), pt(t1, p0), pt(t1, p1), pt(t0, p1)
            tris += [[a, b, c], [a, c, d]]
    return np.array(tris, dtype=float)


def write_binary_stl(path, tris):
    with open(path, "wb") as f:
        f.write(b"solid but actually binary".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            f.write(struct.pack("<3f", 0, 0, 0))
            for v in t:
                f.write(struct.pack("<3f", *v))
            f.write(b"\0\0")


def write_ascii_stl(path, tris):
    with open(path, "w") as f:
        f.write("solid test\n")
        for t in tris:
            f.write(" facet normal 0 0 0\n  outer loop\n")
            for v in t:
                f.write(f"   vertex {v[0]} {v[1]} {v[2]}\n")
            f.write("  endloop\n endfacet\n")
        f.write("endsolid test\n")


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def run_cli(self, stl, *args):
        out = self.tmp / "out.nbt"
        s2s.main([str(stl), "-o", str(out), *args])
        return read_nbt(gzip.decompress(out.read_bytes()))

    def test_binary_and_ascii_load_identically(self):
        tris = cube_tris()
        write_binary_stl(self.tmp / "b.stl", tris)
        write_ascii_stl(self.tmp / "a.stl", tris)
        np.testing.assert_allclose(s2s.load_stl(self.tmp / "b.stl"), tris)
        np.testing.assert_allclose(s2s.load_stl(self.tmp / "a.stl"), tris)

    def test_solid_cube(self):
        write_binary_stl(self.tmp / "c.stl", cube_tris())
        nbt = self.run_cli(self.tmp / "c.stl", "--size", "10")
        self.assertEqual(nbt["size"], [10, 10, 10])
        self.assertEqual(len(nbt["blocks"]), 1000)
        self.assertEqual(nbt["palette"], [{"Name": "minecraft:white_concrete"}])
        self.assertEqual(nbt["DataVersion"], 3955)

    def test_hollow_cube(self):
        write_binary_stl(self.tmp / "c.stl", cube_tris())
        nbt = self.run_cli(self.tmp / "c.stl", "--size", "10", "--hollow")
        self.assertEqual(len(nbt["blocks"]), 1000 - 8 ** 3)

    def test_sphere_fill_with_separate_block(self):
        write_binary_stl(self.tmp / "s.stl", sphere_tris())
        nbt = self.run_cli(self.tmp / "s.stl", "--size", "30",
                           "--block", "glass", "--fill-block", "oak_planks")
        names = [p["Name"] for p in nbt["palette"]]
        self.assertEqual(names, ["minecraft:glass", "minecraft:oak_planks"])
        n = len(nbt["blocks"])
        expected = 4 / 3 * math.pi * 15 ** 3
        self.assertLess(abs(n - expected) / expected, 0.15)
        # Centre block must be interior fill.
        centre = [b for b in nbt["blocks"] if b["pos"] == [15, 15, 15]]
        self.assertEqual(centre[0]["state"], 1)

    def test_z_up_becomes_y_up(self):
        # A tall thin box along CAD Z must end up tall along Minecraft Y.
        tris = cube_tris(1.0) * np.array([2, 4, 20])
        write_binary_stl(self.tmp / "t.stl", tris)
        nbt = self.run_cli(self.tmp / "t.stl", "--scale", "1")
        self.assertEqual(nbt["size"], [2, 20, 4])

    def test_block_state_properties(self):
        entry = s2s.parse_block_state("oak_stairs[facing=north,half=top]")
        self.assertEqual(entry["Name"], (s2s.TAG_STRING, "minecraft:oak_stairs"))
        self.assertEqual(set(entry["Properties"][1]), {"facing", "half"})


if __name__ == "__main__":
    unittest.main()
