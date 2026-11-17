import base64, json, os, sys
glb, tpl, outdir, title = sys.argv[1:5]
data = open(glb, 'rb').read(); CH = 4_500_000          # 4.5 MB -> 6 MB base64 per part (GitHub web upload limits)
os.makedirs(outdir, exist_ok=True); parts = []
for i in range(0, len(data), CH):
    b = data[i:i + CH]; s = base64.b64encode(b).decode(); n = len(parts)
    open(f'{outdir}/model-{n}.txt', 'w').write(s); parts.append([f'model-{n}.txt', len(s), len(b)])
h = open(tpl, encoding='utf-8').read().replace('__PARTS__', json.dumps(parts)).replace('__TITLE__', title)
open(f'{outdir}/index.html', 'w', encoding='utf-8').write(h)
print(json.dumps(parts))
