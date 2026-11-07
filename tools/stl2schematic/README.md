# stl2schematic

Converts `.stl` files (from Fusion 360, Onshape, SolidWorks, FreeCAD, Blender…)
into schematics that the **Create** mod's Schematic Table and Schematicannon can load.

## Setup

Requires Python 3.9+ and numpy:

```
pip install numpy
```

## Usage

**Easiest:** just run the script with no arguments (double-click it, or press Run in
VS Code). It opens a file picker and asks for the size, hollow/solid, the block, and where
to save.

**Command line:**

```
python stl2schematic.py my_plane.stl --size 80
```

This writes `my_plane.nbt` next to the STL. Copy it to `.minecraft/schematics/`
(or `<instance>/schematics/` in a launcher like Prism/CurseForge). Then pick it
in the Schematic Table, write it onto an empty schematic, and place it with a
Schematicannon.

### Options

| Option | Default | What it does |
|---|---|---|
| `--size N` | 64 | Longest side of the model becomes N blocks |
| `--scale S` | – | Blocks per model unit instead of `--size` (e.g. `0.1` = 1 block per 10 mm) |
| `--up {x,y,z}` | `z` | Which axis is "up" in your CAD file. Most CAD is Z-up; Blender exports are usually Y-up |
| `--yaw {0,90,180,270}` | 0 | Spin the model around the vertical axis |
| `--block B` | `minecraft:white_concrete` | Block for the outer shell. Block states work: `"oak_stairs[facing=north]"` |
| `--fill-block B` | same as `--block` | Block for the inside |
| `--hollow` | off | Only build the shell. Uses far fewer blocks, which is usually what you want for a plane |
| `-o FILE` | `<input>.nbt` | Output path |
| `--mc-version` | `1.21.1` | Data version written to the file (`1.20.1`, `1.21`, `1.21.1`) |

Examples:

```
# Hollow 120-block-long airliner with a glass skin
python stl2schematic.py airliner.stl --size 120 --hollow --block glass

# Fighter modelled in mm, 1 block = 50 mm, gray shell, iron interior
python stl2schematic.py fighter.stl --scale 0.02 --block gray_concrete --fill-block iron_block
```

## How it works

1. Reads binary or ASCII STL.
2. Rotates from your CAD's up axis to Minecraft's Y-up (rotation only, so left/right
   wings never get mirrored) and scales to blocks.
3. Samples every triangle densely and marks each block it touches, giving a closed shell.
4. Unless `--hollow`, flood-fills from outside the model; anything the flood can't reach
   is the interior and gets filled.
5. Writes a gzipped NBT structure file (the same format Create and structure blocks use).

## Tips and limits

- **Upload size limit:** Create refuses schematics over 256 KB by default. The tool warns
  you when a file is bigger. Either use `--hollow`, lower `--size`, or raise
  `maxTotalSchematicSize` in `create-server.toml` (in `saves/<world>/serverconfig/` for
  single player).
- **Thin parts:** wings or fins thinner than one block still get a single layer of blocks.
  For smoother detail, increase `--size`.
- **Holes in the mesh:** fill needs a closed (watertight) mesh. If a mesh has gaps, the
  flood leaks in and you get just the shell; the tool prints a note when no interior was
  found. Most CAD exports are watertight. If one isn't, fix it in your CAD/slicer or use `--hollow`.
- **Multiple bodies:** each closed body is filled on its own, so assemblies export fine as one STL.
- Overlapping/intersecting bodies are fine.

## Tests

```
python -m unittest test_stl2schematic -v
```
