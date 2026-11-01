"""Build the cat Harry costume pack for the Silent Hill (1999) Android port: cat ears and a swinging cat tail
on Harry, both hair coloured. Every mesh is made here from scratch; no game data is read or included.

usage: cat_harry_pack.py [out.shc]        build the pack (default: halloween.shc)
       cat_harry_pack.py --obj <out_dir>  also write each piece as an OBJ (bone-local, rest pose)

The port loads it from mods/halloween.shc (next to gamedata/). Without the file nothing changes.

Everything is little-endian, PSX model space (Y down, +Z forward, Harry's left = -X; the hips bone is turned
180 degrees about Y, so on it +Z is backward). Layout (version 2):
  header  char[4] "SHCP", u32 version, u32 piece_count, u32 tex_w, u32 tex_h, u8 rgba[w*h*4] (rows top first)
  piece   u32 chara (game Chara_ id), u32 bone, u32 flags (1 = chain, 2 = hide in first person),
          u32 hide_mask (that character's bones not drawn while the piece is), u32 ring_count,
          u32 vert_count, u32 tri_count, f32 stiff_base, f32 stiff_tip, f32 damping, f32 gravity,
          f32 sway_amp, f32 sway_hz,
          ring_count x f32 rest[3] (chain joints, bone-local),
          vert_count x { f32 pos[3], f32 nrm[3], u8 u, u8 v, u8 ring, u8 pad },
          tri_count x u16 idx[3] (PSX order: outward = left-handed in model space, i.e. clockwise on screen;
                                   built right-handed below and flipped on write, so the OBJs stay standard)
  places  u32 count (0 here; the port also supports world props)
Rigid pieces: pos/nrm are bone-local. Chain pieces: pos/nrm are in their ring's frame (Z along the chain,
X = the bone's X with the tangent removed, Y = Z cross X); the port moves the rings with spring physics.
"""
import math, os, struct, sys

MAGIC, VERSION = b"SHCP", 2
TEX = 256
CHAIN, HIDE_FPS = 1, 2

HARRY = 1
HEAD, HIPS = 2, 11

# 32x32 colour swatches; Harry's hair sampled from his own texture (40,32,24 at full shade).
SWATCH = [
    (44, 34, 26),     # 0 Harry's hair
    (150, 96, 100),   # 1 inner ear
]

def sub(a, b): return [x - y for x, y in zip(a, b)]
def add(a, b): return [x + y for x, y in zip(a, b)]
def mul(a, k): return [x * k for x in a]
def dot(a, b): return sum(x * y for x, y in zip(a, b))
def cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
def unit(a):
    l = math.sqrt(dot(a, a))
    return mul(a, 1 / l) if l > 1e-9 else [0.0, 0.0, 0.0]

def uv(sw): return ((sw % 8) * 32 + 16, (sw // 8) * 32 + 16)

class Mesh:
    def __init__(self):
        self.v, self.t = [], []   # v: (pos, nrm, swatch, ring)
    def vert(self, p, n, sw, ring=0):
        self.v.append((list(p), unit(n), sw, ring))
        return len(self.v) - 1
    def tri(self, a, b, c, out, world=None):
        """Order a,b,c so the face normal points along out (world maps a vertex to rest-pose model space)."""
        P = world or (lambda i: self.v[i][0])
        if dot(cross(sub(P(b), P(a)), sub(P(c), P(a))), out) < 0:
            b, c = c, b
        self.t.append((a, b, c))

def frame(T, side=(1.0, 0.0, 0.0)):
    X = unit(sub(side, mul(T, dot(side, T))))
    if dot(X, X) < 0.5:
        X = unit(sub([0.0, 1.0, 0.0], mul(T, T[1])))
    return X, cross(T, X)

def tube(m, centers, radii, sw, segs=8, rigid=True, tip=True, base_cap=False, sw_tip=None, rot=0.0):
    """Rings around a polyline. Rigid: positions in bone space. Chain: ring-frame local (ring index = joint)."""
    n = len(centers)
    tang = [unit(sub(centers[min(i + 1, n - 1)], centers[max(i - 1, 0)])) for i in range(n)]
    frames = [frame(T) for T in tang]
    rings, worldpos = [], {}
    for i in range(n):
        X, Y = frames[i]
        T = tang[i]
        row = []
        s = sw_tip if (sw_tip is not None and i >= n - 2) else sw
        for k in range(segs):
            a = 2 * math.pi * k / segs + rot
            loc = [radii[i] * math.cos(a), radii[i] * math.sin(a), 0.0]
            wp = add(centers[i], add(mul(X, loc[0]), mul(Y, loc[1])))
            if rigid:
                vi = m.vert(wp, sub(wp, centers[i]), s)
            else:
                vi = m.vert(loc, [math.cos(a), math.sin(a), 0.0], s, i)
            worldpos[vi] = wp
            row.append(vi)
        rings.append(row)
    P = lambda i: worldpos[i]
    for i in range(n - 1):
        for k in range(segs):
            a, b = rings[i][k], rings[i][(k + 1) % segs]
            c, d = rings[i + 1][(k + 1) % segs], rings[i + 1][k]
            mid = mul(add(centers[i], centers[i + 1]), 0.5)
            out = sub(mul(add(add(P(a), P(b)), add(P(c), P(d))), 0.25), mid)
            m.tri(a, b, c, out, P)
            m.tri(a, c, d, out, P)
    def cap(i, sign, length):
        X, Y = frames[i]
        T = tang[i]
        s = sw_tip if sw_tip is not None else sw
        ep = add(centers[i], mul(T, sign * length))
        if rigid:
            vi = m.vert(ep, mul(T, sign), s)
        else:
            vi = m.vert([0.0, 0.0, sign * length], [0.0, 0.0, sign], s, i)
        worldpos[vi] = ep
        for k in range(segs):
            m.tri(rings[i][k], rings[i][(k + 1) % segs], vi, mul(T, sign), P)
    if tip:
        cap(n - 1, 1, radii[-1] * 1.1)
    if base_cap:
        cap(0, -1, radii[0] * 0.5)
    return worldpos

def rot_y(v, a):
    c, s = math.cos(a), math.sin(a)
    return [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]]

def rot_z(v, a):
    c, s = math.cos(a), math.sin(a)
    return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]

def cat_ear(m, base, side):
    """Flattened pyramid: hair-coloured shell, pink inner triangle on the front; base sunk into the hair."""
    w, d, h = 7.5, 4.0, 27.0
    def place(p):
        p = rot_z(p, side * math.radians(14))     # lean outward
        p = rot_y(p, -side * math.radians(16))    # turn the opening a little outward
        return add(base, p)
    fl, fr = place([-w, 0, d]), place([w, 0, d])
    bl, br = place([-w * 0.8, 0, -d]), place([w * 0.8, 0, -d])
    ap = place([0, -h, -1.0])
    c = mul(add(add(fl, fr), add(add(bl, br), ap)), 0.2)
    pts = [fl, fr, br, bl]
    for (a, b) in ((0, 1), (1, 2), (2, 3), (3, 0)):
        pa, pb = pts[a], pts[b]
        n = cross(sub(pb, pa), sub(ap, pa))
        if dot(n, sub(mul(add(add(pa, pb), ap), 1 / 3), c)) < 0:
            n = mul(n, -1)
        i0, i1, i2 = m.vert(pa, n, 0), m.vert(pb, n, 0), m.vert(ap, n, 0)
        m.tri(i0, i1, i2, n)
    # inner ear: a smaller triangle just in front of the front face
    n = cross(sub(fr, fl), sub(ap, fl))
    if dot(n, sub(mul(add(add(fl, fr), ap), 1 / 3), c)) < 0:
        n = mul(n, -1)
    mid = mul(add(add(fl, fr), ap), 1 / 3)
    lift = mul(unit(n), 0.9)
    ins = [add(add(mid, mul(sub(p, mid), 0.62)), lift) for p in (fl, fr, ap)]
    ins[0], ins[1] = add(ins[0], mul(sub(ap, mid), -0.12)), add(ins[1], mul(sub(ap, mid), -0.12))
    i0, i1, i2 = (m.vert(p, n, 1) for p in ins)
    m.tri(i0, i1, i2, n)

def tail_rest(anchor, n, seg, a0, a1, sideways=0.0):
    """Polyline from anchor going backward (+Z on the hips bone), pitched from a0 (below horizontal) to a1."""
    pts, p = [list(anchor)], list(anchor)
    for i in range(n):
        f = i / max(n - 1, 1)
        a = math.radians(a0 + (a1 - a0) * f)
        d = [sideways * f, -math.sin(a), math.cos(a)]   # Y down: positive pitch goes up
        p = add(p, mul(unit(d), seg))
        pts.append(p)
    return pts

def cat_tail(m):
    rest = tail_rest([0.0, 4.0, 17.5], 8, 18.0, -62.0, 10.0)   # from just under the jacket, out clear of the seat, tip curls back
    radii = [6.5, 6.2, 5.8, 5.4, 5.0, 4.8, 4.8, 5.0, 4.6]
    tube(m, rest, radii, 0, segs=8, rigid=False, tip=True)
    return rest

def pieces():
    out = []
    # Harry: head top at y -49 (head bone), hair spans x -20..20, z -13..35.
    ears = Mesh()
    for side in (-1, 1):
        cat_ear(ears, [side * 11.0, -36.0, 8.0], side)
    out.append(dict(bone=HEAD, flags=HIDE_FPS, rings=[], mesh=ears, params=(0, 0, 0, 0, 0, 0)))
    # stiff_base, stiff_tip, damping, gravity, sway_amp, sway_hz
    t = Mesh()
    rest = cat_tail(t)
    out.append(dict(bone=HIPS, flags=CHAIN, rings=rest, mesh=t, params=(0.45, 0.10, 0.72, 0.30, 40.0, 0.6)))
    return out

def texture():
    px = bytearray(TEX * TEX * 4)
    for k, c in enumerate(SWATCH):
        x0, y0 = (k % 8) * 32, (k // 8) * 32
        for y in range(y0, y0 + 32):
            for x in range(x0, x0 + 32):
                o = (y * TEX + x) * 4
                px[o:o + 4] = bytes((*c, 255))
    return bytes(px)

def build(path):
    ps = pieces()
    parts = [struct.pack("<4s4I", MAGIC, VERSION, len(ps), TEX, TEX), texture()]
    for p in ps:
        m = p["mesh"]
        parts.append(struct.pack("<7I6f", HARRY, p["bone"], p["flags"], 0, len(p["rings"]),
                                 len(m.v), len(m.t), *p["params"]))
        parts += [struct.pack("<3f", *r) for r in p["rings"]]
        for pos, nrm, sw, ring in m.v:
            u, v = uv(sw)
            parts.append(struct.pack("<6f4B", *pos, *nrm, u, v, ring, 0))
        parts += [struct.pack("<3H", a, c, b) for a, b, c in m.t]
        print(f"  bone {p['bone']:2} {'chain' if p['flags'] & CHAIN else 'rigid'}: {len(m.v)} verts, {len(m.t)} tris")
    parts.append(struct.pack("<I", 0))
    data = b"".join(parts)
    open(path, "wb").write(data)
    print(f"{path}: {len(ps)} pieces, {len(data) / 1e3:.1f} KB")
    return ps

def write_objs(ps, out_dir):
    """Rest pose, bone-local (the tail placed on its rest polyline), with swatch colours as materials."""
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "cat_harry.mtl"), "w").write(
        "".join(f"newmtl s{k}\nKd {c[0] / 255:.3f} {c[1] / 255:.3f} {c[2] / 255:.3f}\n\n" for k, c in enumerate(SWATCH)))
    for p in ps:
        m, rest = p["mesh"], p["rings"]
        frames = []
        for i in range(len(rest)):
            T = unit(sub(rest[min(i + 1, len(rest) - 1)], rest[max(i - 1, 0)]))
            X, Y = frame(T)
            frames.append((X, Y, T))
        lines = ["mtllib cat_harry.mtl"]
        for pos, nrm, sw, ring in m.v:
            if p["flags"] & CHAIN:
                X, Y, T = frames[ring]
                pos = add(rest[ring], add(add(mul(X, pos[0]), mul(Y, pos[1])), mul(T, pos[2])))
            lines.append(f"v {pos[0]:.2f} {-pos[1]:.2f} {-pos[2]:.2f}")
        cur = None
        for a, b, c in m.t:
            sw = m.v[a][2]
            if sw != cur:
                lines.append(f"usemtl s{sw}")
                cur = sw
            lines.append(f"f {a + 1} {b + 1} {c + 1}")
        name = "ears.obj" if p["bone"] == HEAD else "tail.obj"
        open(os.path.join(out_dir, name), "w").write("\n".join(lines) + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--obj":
        os.makedirs(sys.argv[2], exist_ok=True)
        write_objs(build(os.path.join(sys.argv[2], "halloween.shc")), sys.argv[2])
    else:
        build(sys.argv[1] if len(sys.argv) > 1 else "halloween.shc")
