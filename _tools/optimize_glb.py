"""Optimize a SketchUp/SimLab GLB for web viewing (pure numpy, no external tools)."""
import json, struct, sys, collections, hashlib, io
import numpy as np
from PIL import Image

SRC, DST = sys.argv[1], sys.argv[2]
MIN_INST = int(sys.argv[3]) if len(sys.argv) > 3 else 2
BOXIFY_PREFIXES = ('Led screen',)      # heavy decorative parts replaced by their bounding box
BOX_MIN_TRIS = 2000
MAX_TEX = 1024

raw = open(SRC, 'rb').read()
clen, _ = struct.unpack('<II', raw[12:20])
J = json.loads(raw[20:20 + clen])
BIN = memoryview(raw)[20 + clen + 8:]
N, A, BV = J['nodes'], J['accessors'], J['bufferViews']
CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
_cache = {}

def acc(i):
    if i in _cache: return _cache[i]
    a = A[i]; bv = BV[a['bufferView']]
    dt = np.dtype(CT[a['componentType']]); nc = NC[a['type']]
    off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    if (bv.get('byteStride', 0) or dt.itemsize * nc) != dt.itemsize * nc:
        raise RuntimeError('strided accessor not supported')
    arr = np.frombuffer(BIN, dtype=dt, count=a['count'] * nc, offset=off).reshape(a['count'], nc)
    if nc == 1: arr = arr[:, 0]
    _cache[i] = arr; return arr

def node_mat(n):
    if 'matrix' in n: return np.array(n['matrix'], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    t = n.get('translation', [0, 0, 0]); x, y, z, w = n.get('rotation', [0, 0, 0, 1]); s = n.get('scale', [1, 1, 1])
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    m[:3, :3] = R * np.array(s); m[:3, 3] = t
    return m

# ---- collect (primitive key -> world matrices)
uses = collections.defaultdict(list); boxify = set()
def walk(ni, M, simplify=False):
    n = N[ni]; M = M @ node_mat(n)
    simplify = simplify or (n.get('name') or '').startswith(BOXIFY_PREFIXES)
    if 'mesh' in n:
        for p in J['meshes'][n['mesh']]['primitives']:
            at = p['attributes']
            key = (at['POSITION'], at.get('NORMAL'), at.get('TEXCOORD_0'), p['indices'], p.get('material'))
            uses[key].append(M)
            if simplify and A[p['indices']]['count'] // 3 >= BOX_MIN_TRIS: boxify.add(key)
    for c in n.get('children', []): walk(c, M, simplify)
for r in J['scenes'][J.get('scene', 0)]['nodes']: walk(r, np.eye(4))

def box_geom(lo, hi):
    P = []; Nr = []; I = []
    for axis in range(3):
        for sign in (-1, 1):
            u, v = [a for a in range(3) if a != axis]; base = len(P)
            for cu, cv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                pt = np.empty(3); pt[axis] = hi[axis] if sign > 0 else lo[axis]
                pt[u] = hi[u] if cu else lo[u]; pt[v] = hi[v] if cv else lo[v]
                P.append(pt); nn = np.zeros(3); nn[axis] = sign; Nr.append(nn)
            I += [base, base+1, base+2, base, base+2, base+3]
    return np.array(P), np.array(Nr), np.array(I, np.uint32)

def geom(key):
    pos_i, nor_i, uv_i, idx_i, _ = key
    P = acc(pos_i).astype(np.float64)
    if key in boxify:
        Pb, Nb, Ib = box_geom(P.min(0), P.max(0)); return Pb, Nb, None, Ib
    Nr = acc(nor_i).astype(np.float64) if nor_i is not None else None
    UV = acc(uv_i) if uv_i is not None else None
    return P, Nr, UV, acc(idx_i)

def decompose(M):
    """(t, quat xyzw, s) or None if mirrored / sheared."""
    L = M[:3, :3]
    if np.linalg.det(L) <= 0: return None
    s = np.linalg.norm(L, axis=0)
    if np.any(s < 1e-9): return None
    R = L / s
    if np.abs(R.T @ R - np.eye(3)).max() > 1e-3: return None
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2; w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S; y = (R[0, 2] - R[2, 0]) / S; z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S; x = 0.25 * S; y = (R[0, 1] + R[1, 0]) / S; z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S; x = (R[0, 1] + R[1, 0]) / S; y = 0.25 * S; z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S; x = (R[0, 2] + R[2, 0]) / S; y = (R[1, 2] + R[2, 1]) / S; z = 0.25 * S
    q = np.array([x, y, z, w]); q /= np.linalg.norm(q)
    return M[:3, 3].copy(), q, s

# ---- materials
mats = []; textured = []
for m in J['materials']:
    m = json.loads(json.dumps(m)); m.pop('extensions', None)
    pbr = m.setdefault('pbrMetallicRoughness', {})
    pbr['metallicFactor'] = 0.0; pbr['roughnessFactor'] = 0.75
    alpha = pbr.get('baseColorFactor', [1, 1, 1, 1])[3]
    tex = pbr.get('baseColorTexture')
    png = tex is not None and J['images'][J['textures'][tex['index']]['source']].get('mimeType') == 'image/png'
    if m.get('alphaMode') == 'BLEND' or alpha < 0.999: m['alphaMode'] = 'BLEND'
    elif png: m['alphaMode'] = 'MASK'; m['alphaCutoff'] = 0.5
    else: m['alphaMode'] = 'OPAQUE'; m.pop('alphaCutoff', None)
    m['doubleSided'] = True
    mats.append(m); textured.append(tex is not None)

# ---- output buffer
out = bytearray(); bviews = []; accs = []
def add_bv(data, target=None, stride=None):
    while len(out) % 4: out.append(0)
    bv = {'buffer': 0, 'byteOffset': len(out), 'byteLength': len(data)}
    if target: bv['target'] = target
    if stride: bv['byteStride'] = stride
    out.extend(data); bviews.append(bv); return len(bviews) - 1
def add_acc(bvi, ctype, count, typ, normalized=False, mn=None, mx=None):
    a = {'bufferView': bvi, 'componentType': ctype, 'count': int(count), 'type': typ}
    if normalized: a['normalized'] = True
    if mn is not None:
        a['min'] = [float(v) if ctype == 5126 else int(v) for v in mn]
        a['max'] = [float(v) if ctype == 5126 else int(v) for v in mx]
    accs.append(a); return len(accs) - 1

# ---- images: downscale + dedupe
images = []; img_remap = {}; by_hash = {}
for idx, im in enumerate(J['images']):
    bv = BV[im['bufferView']]
    data = bytes(BIN[bv.get('byteOffset', 0): bv.get('byteOffset', 0) + bv['byteLength']])
    h = hashlib.sha1(data).hexdigest()
    if h in by_hash: img_remap[idx] = by_hash[h]; continue
    mime = im['mimeType']; pic = Image.open(io.BytesIO(data))
    if max(pic.size) > MAX_TEX:
        k = MAX_TEX / max(pic.size)
        pic = pic.resize((max(1, round(pic.size[0]*k)), max(1, round(pic.size[1]*k))), Image.LANCZOS)
        buf = io.BytesIO()
        if mime == 'image/png': pic.save(buf, 'PNG', optimize=True)
        else: pic.convert('RGB').save(buf, 'JPEG', quality=85, optimize=True)
        data = buf.getvalue()
    images.append({'bufferView': add_bv(data), 'mimeType': mime})
    by_hash[h] = len(images) - 1; img_remap[idx] = len(images) - 1

def quant_pos(P, lo, hi):
    span = np.maximum(hi - lo, 1e-6)
    q = np.round((P - lo) / span * 65535).clip(0, 65535).astype(np.uint16)
    buf = np.zeros((len(q), 4), np.uint16); buf[:, :3] = q          # stride 8
    return buf
def pack_normals(Nr):
    n = Nr / np.maximum(np.linalg.norm(Nr, axis=1, keepdims=True), 1e-9)
    buf = np.zeros((len(n), 4), np.int8); buf[:, :3] = np.round(n * 127).clip(-127, 127)
    return buf
def write_prim(P, Nr, UV, I, mat, lo, hi):
    pq = quant_pos(P, lo, hi)
    attrs = {'POSITION': add_acc(add_bv(pq.tobytes(), 34962, 8), 5123, len(P), 'VEC3', False, pq[:, :3].min(0), pq[:, :3].max(0))}
    if Nr is not None:
        attrs['NORMAL'] = add_acc(add_bv(pack_normals(Nr).tobytes(), 34962, 4), 5120, len(P), 'VEC3', True)
    if UV is not None:
        attrs['TEXCOORD_0'] = add_acc(add_bv(UV.astype(np.float32).tobytes(), 34962), 5126, len(P), 'VEC2')
    if len(P) <= 65535: ib = I.astype(np.uint16).tobytes(); ct = 5123
    else: ib = I.astype(np.uint32).tobytes(); ct = 5125
    prim = {'attributes': attrs, 'indices': add_acc(add_bv(ib, 34963), ct, len(I), 'SCALAR')}
    if mat is not None: prim['material'] = mat          # never write material: null
    return prim
span_of = lambda lo, hi: np.maximum(hi - lo, 1e-6) / 65535.0

# ---- instancing (mirrored copies get their own mirrored geometry)
nodes_out = []; meshes_out = []; flat = collections.defaultdict(list)
n_inst_groups = 0; n_inst = 0
MIRROR = np.diag([-1.0, 1.0, 1.0, 1.0]); groups = []
for key, Ms in uses.items():
    pos_m = [M for M in Ms if np.linalg.det(M[:3, :3]) > 0]
    neg_m = [M @ MIRROR for M in Ms if np.linalg.det(M[:3, :3]) <= 0]
    if pos_m: groups.append((key, False, pos_m))
    if neg_m: groups.append((key, True, neg_m))
for key, mirrored, Ms in groups:
    mat = key[4]
    if len(Ms) >= MIN_INST:
        dec = [decompose(M) for M in Ms]
        good = [(M, d) for M, d in zip(Ms, dec) if d is not None]
        bad = [M for M, d in zip(Ms, dec) if d is None]
        if len(good) >= MIN_INST:
            for M in bad: flat[mat].append((key, M @ MIRROR if mirrored else M))
            P, Nr, UVs, I = geom(key)
            if mirrored:
                P = P * [-1, 1, 1]
                if Nr is not None: Nr = Nr * [-1, 1, 1]
                I = I.reshape(-1, 3)[:, ::-1].reshape(-1)
            lo, hi = P.min(0), P.max(0)
            UV = None
            if mat is not None and textured[mat]:
                UV = UVs if UVs is not None else np.zeros((len(P), 2), np.float32)
            meshes_out.append({'primitives': [write_prim(P, Nr, UV, I, mat, lo, hi)]})
            span = span_of(lo, hi); T = []; Q = []; S = []
            for M, (t, q, s) in good:          # bake dequantization into instance TRS
                T.append(t + M[:3, :3] @ lo); Q.append(q); S.append(s * span)
            T = np.array(T, np.float32); Q = np.array(Q, np.float32); S = np.array(S, np.float32)
            ext = {'attributes': {
                'TRANSLATION': add_acc(add_bv(T.tobytes()), 5126, len(T), 'VEC3'),
                'ROTATION': add_acc(add_bv(Q.tobytes()), 5126, len(Q), 'VEC4'),
                'SCALE': add_acc(add_bv(S.tobytes()), 5126, len(S), 'VEC3')}}
            nodes_out.append({'mesh': len(meshes_out) - 1, 'extensions': {'EXT_mesh_gpu_instancing': ext}})
            n_inst_groups += 1; n_inst += len(good)
            continue
    for M in Ms: flat[mat].append((key, M @ MIRROR if mirrored else M))

# ---- flattened, merged per material
allP_lo = np.full(3, np.inf); allP_hi = np.full(3, -np.inf); baked = collections.defaultdict(list)
for mat, items in flat.items():
    use_uv = mat is not None and textured[mat]
    for key, M in items:
        P, Nsrc, UVs, I = geom(key)
        Pw = P @ M[:3, :3].T + M[:3, 3]
        if Nsrc is not None: Nw = Nsrc @ np.linalg.inv(M[:3, :3])
        else: Nw = np.zeros_like(Pw); Nw[:, 1] = 1
        I = I.reshape(-1, 3)
        if np.linalg.det(M[:3, :3]) < 0: I = I[:, ::-1]
        UV = (UVs if UVs is not None else np.zeros((len(P), 2), np.float32)) if use_uv else None
        baked[mat].append((Pw, Nw, UV, I))
        allP_lo = np.minimum(allP_lo, Pw.min(0)); allP_hi = np.maximum(allP_hi, Pw.max(0))
flat_prims = []
def emit(chunk, mat):
    P = np.concatenate([c[0] for c in chunk]); Nr = np.concatenate([c[1] for c in chunk])
    UV = np.concatenate([c[2] for c in chunk]) if chunk[0][2] is not None else None
    Is = []; base = 0
    for c in chunk: Is.append(c[3].reshape(-1) + base); base += len(c[0])
    flat_prims.append(write_prim(P, Nr, UV, np.concatenate(Is), mat, allP_lo, allP_hi))
for mat, items in baked.items():
    chunk = []; cv = 0
    for it in items:
        if chunk and cv + len(it[0]) > 65535: emit(chunk, mat); chunk = []; cv = 0
        chunk.append(it); cv += len(it[0])
    if chunk: emit(chunk, mat)
meshes_out.append({'primitives': flat_prims})
nodes_out.append({'name': 'merged', 'mesh': len(meshes_out) - 1,
                  'translation': [float(v) for v in allP_lo], 'scale': [float(v) for v in span_of(allP_lo, allP_hi)]})

# ---- assemble GLB
G = {'asset': {'version': '2.0', 'generator': 'claude-optimize-glb'},
     'extensionsUsed': ['KHR_mesh_quantization', 'EXT_mesh_gpu_instancing'],
     'extensionsRequired': ['KHR_mesh_quantization', 'EXT_mesh_gpu_instancing'],
     'scene': 0, 'scenes': [{'nodes': [len(nodes_out)]}],
     'nodes': nodes_out + [{'name': 'root', 'children': list(range(len(nodes_out)))}],
     'meshes': meshes_out, 'materials': mats,
     'textures': [{'source': img_remap[t['source']]} for t in J['textures']],
     'images': images, 'accessors': accs, 'bufferViews': bviews, 'buffers': [{'byteLength': len(out)}]}
js = json.dumps(G, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
js += b' ' * ((4 - len(js) % 4) % 4)
while len(out) % 4: out.append(0)
total = 12 + 8 + len(js) + 8 + len(out)
with open(DST, 'wb') as f:
    f.write(struct.pack('<4sII', b'glTF', 2, total))
    f.write(struct.pack('<I4s', len(js), b'JSON')); f.write(js)
    f.write(struct.pack('<I4s', len(out), b'BIN\x00')); f.write(out)
print(f'instanced groups {n_inst_groups} ({n_inst} inst), merged prims {len(flat_prims)}, draws ~{len(flat_prims)+n_inst_groups}')
print(f'bounds {allP_lo.round(2)} .. {allP_hi.round(2)}; boxified {len(boxify)}; images {len(J["images"])} -> {len(images)}; {total/1e6:.1f} MB')
