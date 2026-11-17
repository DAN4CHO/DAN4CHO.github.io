// ===== GLB optimizer for web viewing: JS port of tools/optimize_glb.py =====
// optimizeGLB(arrayBuffer, { resizeImage, log }) -> { glb: Uint8Array, stats }
// resizeImage(bytes: Uint8Array, mime: string, maxSide: number) -> Promise<Uint8Array>

const GLB_CT = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
const GLB_NC = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };
const BOXIFY_PREFIXES = ['Led screen'];   // heavy decorative parts replaced by their bounding box
const BOX_MIN_TRIS = 2000;
const MAX_TEX = 1024;
const MIN_INST = 2;

function parseGLB(ab) {
  const dv = new DataView(ab);
  if (ab.byteLength < 20 || dv.getUint32(0, true) !== 0x46546C67) throw new Error('это не GLB-файл (нужен экспорт GLTF Binary *.glb)');
  const jlen = dv.getUint32(12, true);
  const J = JSON.parse(new TextDecoder().decode(new Uint8Array(ab, 20, jlen)));
  const off = 20 + jlen;
  let BIN = new Uint8Array(0);
  if (off + 8 <= ab.byteLength) BIN = new Uint8Array(ab, off + 8, dv.getUint32(off, true));
  return { J, BIN };
}

// 4x4 matrices: Float64Array(16), row-major M[r*4+c]
function nodeMat(n) {
  const m = new Float64Array(16);
  if (n.matrix) { for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) m[r * 4 + c] = n.matrix[c * 4 + r]; return m; }
  const t = n.translation || [0, 0, 0], q = n.rotation || [0, 0, 0, 1], s = n.scale || [1, 1, 1];
  const [x, y, z, w] = q;
  const R = [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
             2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
             2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) m[r * 4 + c] = R[r * 3 + c] * s[c];
  m[3] = t[0]; m[7] = t[1]; m[11] = t[2]; m[15] = 1;
  return m;
}
function mul4(A, B) {
  const C = new Float64Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    let v = 0; for (let k = 0; k < 4; k++) v += A[r * 4 + k] * B[k * 4 + c]; C[r * 4 + c] = v;
  }
  return C;
}
const IDENT = (() => { const m = new Float64Array(16); m[0] = m[5] = m[10] = m[15] = 1; return m; })();
function det3(M) {
  const a = M[0], b = M[1], c = M[2], d = M[4], e = M[5], f = M[6], g = M[8], h = M[9], i = M[10];
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}
function inv3(M) {      // inverse of the upper-left 3x3, row-major 9
  const a = M[0], b = M[1], c = M[2], d = M[4], e = M[5], f = M[6], g = M[8], h = M[9], i = M[10];
  const D = det3(M) || 1e-30;
  return [(e * i - f * h) / D, (c * h - b * i) / D, (b * f - c * e) / D,
          (f * g - d * i) / D, (a * i - c * g) / D, (c * d - a * f) / D,
          (d * h - e * g) / D, (b * g - a * h) / D, (a * e - b * d) / D];
}
function mirrorCol(M) { const C = M.slice(); C[0] = -C[0]; C[4] = -C[4]; C[8] = -C[8]; C[12] = -C[12]; return C; }

function decompose(M) {          // [t, quat xyzw, s] or null if mirrored / sheared
  if (det3(M) <= 0) return null;
  const s = [0, 1, 2].map(c => Math.hypot(M[c], M[4 + c], M[8 + c]));
  if (s.some(v => v < 1e-9)) return null;
  const R = [];                                       // R[r][c]
  for (let r = 0; r < 3; r++) R.push([0, 1, 2].map(c => M[r * 4 + c] / s[c]));
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let v = 0; for (let k = 0; k < 3; k++) v += R[k][i] * R[k][j];
    if (Math.abs(v - (i === j ? 1 : 0)) > 1e-3) return null;
  }
  const tr = R[0][0] + R[1][1] + R[2][2];
  let x, y, z, w, S;
  if (tr > 0) {
    S = Math.sqrt(tr + 1) * 2; w = 0.25 * S;
    x = (R[2][1] - R[1][2]) / S; y = (R[0][2] - R[2][0]) / S; z = (R[1][0] - R[0][1]) / S;
  } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
    S = Math.sqrt(1 + R[0][0] - R[1][1] - R[2][2]) * 2;
    w = (R[2][1] - R[1][2]) / S; x = 0.25 * S; y = (R[0][1] + R[1][0]) / S; z = (R[0][2] + R[2][0]) / S;
  } else if (R[1][1] > R[2][2]) {
    S = Math.sqrt(1 + R[1][1] - R[0][0] - R[2][2]) * 2;
    w = (R[0][2] - R[2][0]) / S; x = (R[0][1] + R[1][0]) / S; y = 0.25 * S; z = (R[1][2] + R[2][1]) / S;
  } else {
    S = Math.sqrt(1 + R[2][2] - R[0][0] - R[1][1]) * 2;
    w = (R[1][0] - R[0][1]) / S; x = (R[0][2] + R[2][0]) / S; y = (R[1][2] + R[2][1]) / S; z = 0.25 * S;
  }
  const qn = Math.hypot(x, y, z, w);
  return [[M[3], M[7], M[11]], [x / qn, y / qn, z / qn, w / qn], s];
}

function boxGeom(lo, hi) {
  const P = [], N = [], I = [];
  for (let axis = 0; axis < 3; axis++) for (const sign of [-1, 1]) {
    const [u, v] = [0, 1, 2].filter(a => a !== axis); const base = P.length / 3;
    for (const [cu, cv] of [[0, 0], [1, 0], [1, 1], [0, 1]]) {
      const pt = [0, 0, 0]; pt[axis] = sign > 0 ? hi[axis] : lo[axis];
      pt[u] = cu ? hi[u] : lo[u]; pt[v] = cv ? hi[v] : lo[v];
      P.push(...pt); const nn = [0, 0, 0]; nn[axis] = sign; N.push(...nn);
    }
    I.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return { P: Float64Array.from(P), N: Float64Array.from(N), UV: null, I: Uint32Array.from(I), nv: P.length / 3 };
}

function minmax3(P) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < P.length; i += 3) for (let k = 0; k < 3; k++) {
    const v = P[i + k]; if (v < lo[k]) lo[k] = v; if (v > hi[k]) hi[k] = v;
  }
  return [lo, hi];
}
const spanOf = (lo, hi) => [0, 1, 2].map(k => Math.max(hi[k] - lo[k], 1e-6) / 65535);
const roundHalfEven = v => { const r = Math.round(v); return (Math.abs(v % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r; };

async function sha1hex(u8) {
  const d = new Uint8Array(await crypto.subtle.digest('SHA-1', u8));
  return Array.from(d, b => b.toString(16).padStart(2, '0')).join('');
}

async function optimizeGLB(ab, opts = {}) {
  const log = opts.log || (() => {});
  const resizeImage = opts.resizeImage || (async d => d);
  const { J, BIN } = parseGLB(ab);
  const N = J.nodes || [], A = J.accessors || [], BV = J.bufferViews || [];
  const IMGS = J.images || [], TEXS = J.textures || [], MATS = J.materials || [], MESHES = J.meshes || [];

  const cache = new Map();
  const acc = i => {
    if (cache.has(i)) return cache.get(i);
    const a = A[i], bv = BV[a.bufferView];
    const T = GLB_CT[a.componentType], nc = GLB_NC[a.type], isz = T.BYTES_PER_ELEMENT;
    const start = BIN.byteOffset + (bv.byteOffset || 0) + (a.byteOffset || 0);
    const stride = bv.byteStride || isz * nc;
    let arr;
    if (stride === isz * nc && start % isz === 0) arr = new T(BIN.buffer, start, a.count * nc);
    else {
      arr = new T(a.count * nc); const src = new Uint8Array(BIN.buffer), dst = new Uint8Array(arr.buffer), row = isz * nc;
      for (let k = 0; k < a.count; k++) dst.set(src.subarray(start + k * stride, start + k * stride + row), k * row);
    }
    cache.set(i, arr); return arr;
  };

  // ---- collect (primitive key -> world matrices)
  const uses = new Map(), keyInfo = new Map(), boxify = new Set();
  let srcTris = 0;
  const walk = (ni, M, simplify) => {
    const n = N[ni]; M = mul4(M, nodeMat(n));
    simplify = simplify || BOXIFY_PREFIXES.some(p => (n.name || '').startsWith(p));
    if (n.mesh !== undefined) for (const p of MESHES[n.mesh].primitives) {
      if ((p.mode ?? 4) !== 4) continue;                     // triangles only
      const at = p.attributes; if (at.POSITION === undefined) continue;
      const k = [at.POSITION, at.NORMAL ?? null, at.TEXCOORD_0 ?? null, p.indices ?? null, p.material ?? null];
      const ks = k.join('|');
      if (!uses.has(ks)) { uses.set(ks, []); keyInfo.set(ks, k); }
      uses.get(ks).push(M);
      const tris = Math.floor((p.indices !== undefined ? A[p.indices].count : A[at.POSITION].count) / 3);
      srcTris += tris;
      if (simplify && tris >= BOX_MIN_TRIS) boxify.add(ks);
    }
    for (const c of n.children || []) walk(c, M, simplify);
  };
  const scene = (J.scenes || [])[J.scene || 0];
  if (!scene) throw new Error('в файле нет сцены');
  for (const r of scene.nodes || []) walk(r, IDENT, false);
  log(`Объектов в модели: ${N.length}, уникальных деталей: ${uses.size}`);

  const geom = ks => {
    const [pi, nI, uI, iI] = keyInfo.get(ks);
    const Pf = acc(pi);
    if (boxify.has(ks)) { const [lo, hi] = minmax3(Pf); return boxGeom(lo, hi); }
    const nv = Pf.length / 3;
    let I;
    if (iI !== null) I = acc(iI); else { I = new Uint32Array(nv); for (let i = 0; i < nv; i++) I[i] = i; }
    return { P: Float64Array.from(Pf), N: nI !== null ? Float64Array.from(acc(nI)) : null,
             UV: uI !== null ? acc(uI) : null, I, nv };
  };

  // ---- materials
  const mats = [], textured = [];
  for (const m0 of MATS) {
    const m = JSON.parse(JSON.stringify(m0)); delete m.extensions;
    const pbr = m.pbrMetallicRoughness || (m.pbrMetallicRoughness = {});
    pbr.metallicFactor = 0.0; pbr.roughnessFactor = 0.75;
    const alpha = (pbr.baseColorFactor || [1, 1, 1, 1])[3];
    const tex = pbr.baseColorTexture;
    const png = tex !== undefined && IMGS[TEXS[tex.index].source].mimeType === 'image/png';
    if (m.alphaMode === 'BLEND' || alpha < 0.999) m.alphaMode = 'BLEND';
    else if (png) { m.alphaMode = 'MASK'; m.alphaCutoff = 0.5; }
    else { m.alphaMode = 'OPAQUE'; delete m.alphaCutoff; }
    m.doubleSided = true;
    mats.push(m); textured.push(tex !== undefined);
  }

  // ---- output buffer
  const chunks = []; let outLen = 0; const bviews = [], accs = [];
  const asU8 = t => new Uint8Array(t.buffer, t.byteOffset, t.byteLength);
  const addBV = (data, target, stride) => {
    const pad = (4 - outLen % 4) % 4; if (pad) { chunks.push(new Uint8Array(pad)); outLen += pad; }
    const bv = { buffer: 0, byteOffset: outLen, byteLength: data.byteLength };
    if (target) bv.target = target; if (stride) bv.byteStride = stride;
    chunks.push(asU8(data)); outLen += data.byteLength; bviews.push(bv); return bviews.length - 1;
  };
  const addAcc = (bvi, ctype, count, type, normalized, mn, mx) => {
    const a = { bufferView: bvi, componentType: ctype, count, type };
    if (normalized) a.normalized = true;
    if (mn) { a.min = mn.map(Number); a.max = mx.map(Number); }
    accs.push(a); return accs.length - 1;
  };

  // ---- images: downscale + dedupe
  const images = [], imgRemap = {}, byHash = new Map();
  for (let idx = 0; idx < IMGS.length; idx++) {
    const im = IMGS[idx]; const bv = BV[im.bufferView];
    let data = BIN.slice(bv.byteOffset || 0, (bv.byteOffset || 0) + bv.byteLength);
    const h = await sha1hex(data);
    if (byHash.has(h)) { imgRemap[idx] = byHash.get(h); continue; }
    try { data = await resizeImage(data, im.mimeType, MAX_TEX); } catch (e) { /* keep original */ }
    images.push({ bufferView: addBV(data), mimeType: im.mimeType });
    byHash.set(h, images.length - 1); imgRemap[idx] = images.length - 1;
    if (idx % 10 === 9) log(`Текстуры: ${idx + 1} из ${IMGS.length}`, true);
  }
  if (IMGS.length) log(`Текстуры: ${IMGS.length} → ${images.length} (одинаковые склеены, большие ужаты до ${MAX_TEX} px)`);

  const quantPos = (P, lo, hi) => {
    const span = [0, 1, 2].map(k => Math.max(hi[k] - lo[k], 1e-6));
    const n = P.length / 3, q = new Uint16Array(n * 4);
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) {
      let v = roundHalfEven((P[i * 3 + k] - lo[k]) / span[k] * 65535);
      q[i * 4 + k] = v < 0 ? 0 : v > 65535 ? 65535 : v;
    }
    return q;
  };
  const packNormals = Nr => {
    const n = Nr.length / 3, b = new Int8Array(n * 4);
    for (let i = 0; i < n; i++) {
      const x = Nr[i * 3], y = Nr[i * 3 + 1], z = Nr[i * 3 + 2], l = Math.max(Math.hypot(x, y, z), 1e-9);
      for (let k = 0; k < 3; k++) { const v = roundHalfEven(Nr[i * 3 + k] / l * 127); b[i * 4 + k] = v < -127 ? -127 : v > 127 ? 127 : v; }
    }
    return b;
  };
  const writePrim = (P, Nr, UV, I, mat, lo, hi) => {
    const pq = quantPos(P, lo, hi), n = P.length / 3;
    const mn = [65535, 65535, 65535], mx = [0, 0, 0];
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) { const v = pq[i * 4 + k]; if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v; }
    const attrs = { POSITION: addAcc(addBV(pq, 34962, 8), 5123, n, 'VEC3', false, mn, mx) };
    if (Nr) attrs.NORMAL = addAcc(addBV(packNormals(Nr), 34962, 4), 5120, n, 'VEC3', true);
    if (UV) attrs.TEXCOORD_0 = addAcc(addBV(UV instanceof Float32Array ? UV : Float32Array.from(UV), 34962), 5126, n, 'VEC2');
    const ib = n <= 65535 ? Uint16Array.from(I) : Uint32Array.from(I);
    const prim = { attributes: attrs, indices: addAcc(addBV(ib, 34963), n <= 65535 ? 5123 : 5125, I.length, 'SCALAR') };
    if (mat !== null) prim.material = mat;             // never write material: null
    return prim;
  };

  // ---- instancing (mirrored copies get their own mirrored geometry)
  const nodesOut = [], meshesOut = [], flat = new Map();
  const pushFlat = (mat, item) => { if (!flat.has(mat)) flat.set(mat, []); flat.get(mat).push(item); };
  let nInstGroups = 0, nInst = 0;
  const groups = [];
  for (const [ks, Ms] of uses) {
    const pos = [], neg = [];
    for (const M of Ms) (det3(M) > 0 ? pos.push(M) : neg.push(mirrorCol(M)));
    if (pos.length) groups.push([ks, false, pos]);
    if (neg.length) groups.push([ks, true, neg]);
  }
  for (const [ks, mirrored, Ms] of groups) {
    const mat = keyInfo.get(ks)[4];
    if (Ms.length >= MIN_INST) {
      const good = [], bad = [];
      for (const M of Ms) { const d = decompose(M); d ? good.push([M, d]) : bad.push(M); }
      if (good.length >= MIN_INST) {
        for (const M of bad) pushFlat(mat, [ks, mirrored ? mirrorCol(M) : M]);
        let { P, N: Nr, UV: UVs, I, nv } = geom(ks);
        if (mirrored) {
          for (let i = 0; i < P.length; i += 3) P[i] = -P[i];
          if (Nr) for (let i = 0; i < Nr.length; i += 3) Nr[i] = -Nr[i];
          const J2 = new Uint32Array(I.length);
          for (let t = 0; t < I.length; t += 3) { J2[t] = I[t + 2]; J2[t + 1] = I[t + 1]; J2[t + 2] = I[t]; }
          I = J2;
        }
        const [lo, hi] = minmax3(P);
        let UV = null;
        if (mat !== null && textured[mat]) UV = UVs || new Float32Array(nv * 2);
        meshesOut.push({ primitives: [writePrim(P, Nr, UV, I, mat, lo, hi)] });
        const span = spanOf(lo, hi), n = good.length;
        const T = new Float32Array(n * 3), Q = new Float32Array(n * 4), S = new Float32Array(n * 3);
        good.forEach(([M, [t, q, s]], j) => {         // bake dequantization into instance TRS
          for (let r = 0; r < 3; r++) T[j * 3 + r] = t[r] + M[r * 4] * lo[0] + M[r * 4 + 1] * lo[1] + M[r * 4 + 2] * lo[2];
          Q.set(q, j * 4);
          for (let k = 0; k < 3; k++) S[j * 3 + k] = s[k] * span[k];
        });
        const ext = { attributes: {
          TRANSLATION: addAcc(addBV(T), 5126, n, 'VEC3'),
          ROTATION: addAcc(addBV(Q), 5126, n, 'VEC4'),
          SCALE: addAcc(addBV(S), 5126, n, 'VEC3') } };
        nodesOut.push({ mesh: meshesOut.length - 1, extensions: { EXT_mesh_gpu_instancing: ext } });
        nInstGroups++; nInst += n;
        continue;
      }
    }
    for (const M of Ms) pushFlat(mat, [ks, mirrored ? mirrorCol(M) : M]);
  }
  log(`Повторяющиеся детали: ${nInstGroups} групп, ${nInst} копий`);

  // ---- flattened, merged per material
  const allLo = [Infinity, Infinity, Infinity], allHi = [-Infinity, -Infinity, -Infinity];
  const baked = new Map();
  for (const [mat, items] of flat) {
    const useUV = mat !== null && textured[mat];
    const list = []; baked.set(mat, list);
    for (const [ks, M] of items) {
      const { P, N: Nsrc, UV: UVs, I, nv } = geom(ks);
      const Pw = new Float64Array(nv * 3), Nw = new Float64Array(nv * 3);
      for (let i = 0; i < nv; i++) {
        const x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2];
        for (let r = 0; r < 3; r++) {
          const v = M[r * 4] * x + M[r * 4 + 1] * y + M[r * 4 + 2] * z + M[r * 4 + 3];
          Pw[i * 3 + r] = v; if (v < allLo[r]) allLo[r] = v; if (v > allHi[r]) allHi[r] = v;
        }
      }
      if (Nsrc) {
        const Mi = inv3(M);                    // n' = n @ inv(L)  (inverse-transpose)
        for (let i = 0; i < nv; i++) {
          const x = Nsrc[i * 3], y = Nsrc[i * 3 + 1], z = Nsrc[i * 3 + 2];
          for (let c = 0; c < 3; c++) Nw[i * 3 + c] = x * Mi[c] + y * Mi[3 + c] + z * Mi[6 + c];
        }
      } else for (let i = 0; i < nv; i++) Nw[i * 3 + 1] = 1;
      const Iw = new Uint32Array(I.length);
      if (det3(M) < 0) for (let t = 0; t < I.length; t += 3) { Iw[t] = I[t + 2]; Iw[t + 1] = I[t + 1]; Iw[t + 2] = I[t]; }
      else Iw.set(I);
      const UV = useUV ? (UVs || new Float32Array(nv * 2)) : null;
      list.push([Pw, Nw, UV, Iw]);
    }
  }
  const flatPrims = [];
  const emit = (chunk, mat) => {
    let nv = 0, ni = 0; for (const c of chunk) { nv += c[0].length / 3; ni += c[3].length; }
    const P = new Float64Array(nv * 3), Nr = new Float64Array(nv * 3), I = new Uint32Array(ni);
    const UV = chunk[0][2] ? new Float32Array(nv * 2) : null;
    let vo = 0, io = 0;
    for (const [p, n, uv, idx] of chunk) {
      P.set(p, vo * 3); Nr.set(n, vo * 3); if (UV) UV.set(uv, vo * 2);
      for (let k = 0; k < idx.length; k++) I[io + k] = idx[k] + vo;
      vo += p.length / 3; io += idx.length;
    }
    flatPrims.push(writePrim(P, Nr, UV, I, mat, allLo, allHi));
  };
  for (const [mat, items] of baked) {
    let chunk = [], cv = 0;
    for (const it of items) {
      const n = it[0].length / 3;
      if (chunk.length && cv + n > 65535) { emit(chunk, mat); chunk = []; cv = 0; }
      chunk.push(it); cv += n;
    }
    if (chunk.length) emit(chunk, mat);
  }
  if (flatPrims.length) {
    meshesOut.push({ primitives: flatPrims });
    nodesOut.push({ name: 'merged', mesh: meshesOut.length - 1, translation: allLo.slice(), scale: spanOf(allLo, allHi) });
  }

  // ---- assemble GLB
  const G = { asset: { version: '2.0', generator: 'claude-optimize-glb-js' },
    extensionsUsed: ['KHR_mesh_quantization', 'EXT_mesh_gpu_instancing'],
    extensionsRequired: ['KHR_mesh_quantization', 'EXT_mesh_gpu_instancing'],
    scene: 0, scenes: [{ nodes: [nodesOut.length] }],
    nodes: nodesOut.concat([{ name: 'root', children: nodesOut.map((_, i) => i) }]),
    meshes: meshesOut, materials: mats,
    textures: TEXS.map(t => ({ source: imgRemap[t.source] })),
    images, accessors: accs, bufferViews: bviews, buffers: [{ byteLength: 0 }] };
  { const pad = (4 - outLen % 4) % 4; if (pad) { chunks.push(new Uint8Array(pad)); outLen += pad; } }
  G.buffers[0].byteLength = outLen;
  if (!images.length) { delete G.images; delete G.textures; }
  let js = new TextEncoder().encode(JSON.stringify(G));
  const jpad = (4 - js.length % 4) % 4;
  const total = 12 + 8 + js.length + jpad + 8 + outLen;
  const out = new Uint8Array(total), dv = new DataView(out.buffer);
  dv.setUint32(0, 0x46546C67, true); dv.setUint32(4, 2, true); dv.setUint32(8, total, true);
  dv.setUint32(12, js.length + jpad, true); dv.setUint32(16, 0x4E4F534A, true);
  out.set(js, 20); out.fill(0x20, 20 + js.length, 20 + js.length + jpad);
  let o = 20 + js.length + jpad;
  dv.setUint32(o, outLen, true); dv.setUint32(o + 4, 0x004E4942, true); o += 8;
  for (const c of chunks) { out.set(c, o); o += c.byteLength; }

  let outTris = 0;
  for (const nd of nodesOut) {
    const mult = nd.extensions ? accs[nd.extensions.EXT_mesh_gpu_instancing.attributes.TRANSLATION].count : 1;
    for (const p of meshesOut[nd.mesh].primitives) outTris += accs[p.indices].count / 3 * mult;
  }
  return { glb: out, stats: { srcBytes: ab.byteLength, outBytes: total, srcTris, outTris,
    draws: flatPrims.length + nInstGroups, instGroups: nInstGroups, inst: nInst, boxified: boxify.size,
    imagesIn: IMGS.length, imagesOut: images.length } };
}

// ===== packing: base64 parts + index.html + zip (store) =====
const B64CH = new TextEncoder().encode('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/');
function b64encode(u8) {
  const n = u8.length, out = new Uint8Array(Math.ceil(n / 3) * 4); let i = 0, j = 0;
  for (; i + 2 < n; i += 3) {
    const v = (u8[i] << 16) | (u8[i + 1] << 8) | u8[i + 2];
    out[j++] = B64CH[v >> 18]; out[j++] = B64CH[(v >> 12) & 63]; out[j++] = B64CH[(v >> 6) & 63]; out[j++] = B64CH[v & 63];
  }
  if (i < n) {
    const v = (u8[i] << 16) | ((i + 1 < n ? u8[i + 1] : 0) << 8);
    out[j++] = B64CH[v >> 18]; out[j++] = B64CH[(v >> 12) & 63];
    out[j++] = i + 1 < n ? B64CH[(v >> 6) & 63] : 61; out[j++] = 61;
  }
  return out;
}
const escHtml = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
function buildSite(glb, template, title, CH = 4_500_000) {   // 4.5 MB -> 6 MB base64 per part (browser upload limits)
  const files = [], parts = [];
  for (let i = 0; i < glb.length; i += CH) {
    const b = glb.subarray(i, i + CH), s = b64encode(b), n = parts.length;
    files.push([`model-${n}.txt`, s]); parts.push([`model-${n}.txt`, s.length, b.length]);
  }
  const html = template.split('__PARTS__').join(JSON.stringify(parts)).split('__TITLE__').join(escHtml(title));
  files.push(['index.html', new TextEncoder().encode(html)]);   // last: the page goes live only after all parts
  return files;
}
function batchFiles(files, limit = 18_000_000) {              // upload batches for the GitHub web uploader
  const batches = []; let cur = [], size = 0;
  for (const f of files) {
    if (cur.length && size + f[1].length > limit) { batches.push(cur); cur = []; size = 0; }
    cur.push(f[0]); size += f[1].length;
  }
  if (cur.length) batches.push(cur);
  return batches;
}
const CRC_T = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(u8) { let c = 0xFFFFFFFF; for (let i = 0; i < u8.length; i++) c = CRC_T[(c ^ u8[i]) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; }
function makeZip(entries) {            // entries: [[path, Uint8Array]] -> array of Uint8Array pieces
  const enc = new TextEncoder(), pieces = [], central = []; let off = 0;
  for (const [path, data] of entries) {
    const name = enc.encode(path), crc = crc32(data);
    const h = new Uint8Array(30 + name.length), dv = new DataView(h.buffer);
    dv.setUint32(0, 0x04034b50, true); dv.setUint16(4, 20, true); dv.setUint16(6, 0x0800, true);
    dv.setUint16(8, 0, true); dv.setUint16(10, 0, true); dv.setUint16(12, 0x21, true);
    dv.setUint32(14, crc, true); dv.setUint32(18, data.length, true); dv.setUint32(22, data.length, true);
    dv.setUint16(26, name.length, true); dv.setUint16(28, 0, true); h.set(name, 30);
    pieces.push(h, data);
    const c = new Uint8Array(46 + name.length), cv = new DataView(c.buffer);
    cv.setUint32(0, 0x02014b50, true); cv.setUint16(4, 20, true); cv.setUint16(6, 20, true); cv.setUint16(8, 0x0800, true);
    cv.setUint16(10, 0, true); cv.setUint16(12, 0, true); cv.setUint16(14, 0x21, true);
    cv.setUint32(16, crc, true); cv.setUint32(20, data.length, true); cv.setUint32(24, data.length, true);
    cv.setUint16(28, name.length, true); cv.setUint32(42, off, true); c.set(name, 46);
    central.push(c); off += h.length + data.length;
  }
  const csize = central.reduce((s, c) => s + c.length, 0);
  const e = new Uint8Array(22), ev = new DataView(e.buffer);
  ev.setUint32(0, 0x06054b50, true); ev.setUint16(8, entries.length, true); ev.setUint16(10, entries.length, true);
  ev.setUint32(12, csize, true); ev.setUint32(16, off, true);
  return pieces.concat(central, [e]);
}
const TRANSLIT = { а:'a',б:'b',в:'v',г:'g',д:'d',е:'e',ё:'e',ж:'zh',з:'z',и:'i',й:'y',к:'k',л:'l',м:'m',н:'n',о:'o',п:'p',р:'r',с:'s',т:'t',у:'u',ф:'f',х:'h',ц:'ts',ч:'ch',ш:'sh',щ:'sch',ъ:'',ы:'y',ь:'',э:'e',ю:'yu',я:'ya' };
function slugify(s) {
  return s.toLowerCase().split('').map(ch => TRANSLIT[ch] ?? ch).join('')
    .replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'project';
}
