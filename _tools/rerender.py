"""Re-render a project's index.html with a new template, keeping its PARTS and title."""
import json, re, sys, html
tpl_path, src_index, out_index = sys.argv[1:4]
prefix = sys.argv[4] if len(sys.argv) > 4 else ''
title_suffix = sys.argv[5] if len(sys.argv) > 5 else ''
s = open(src_index, encoding='utf-8').read()
parts = json.loads(re.search(r'const PARTS = (\[.*?\]);', s).group(1))
title = html.unescape(re.search(r'<title>(.*?) · 3D-просмотр</title>', s).group(1))
parts = [[prefix + p[0]] + p[1:] for p in parts]
t = open(tpl_path, encoding='utf-8').read()
t = t.replace('__PARTS__', json.dumps(parts)).replace('__TITLE__', html.escape(title + title_suffix, quote=True))
open(out_index, 'w', encoding='utf-8').write(t)
print(title, len(parts), 'parts')
