"""Rebuild packer/index.html from core.js + packer.src.html + viewer_template.html (run from repo root)."""
import json, os
D = os.path.dirname(os.path.abspath(__file__))
src = open(f'{D}/packer.src.html', encoding='utf-8').read()
core = open(f'{D}/core.js', encoding='utf-8').read()
tpl = open(f'{D}/viewer_template.html', encoding='utf-8').read()
assert '</script' not in core
src = src.replace('/*__CORE__*/', core, 1).replace('__TEMPLATE_JSON__', json.dumps(tpl, ensure_ascii=False).replace('<', '\\u003c'), 1)
open(f'{D}/../packer/index.html', 'w', encoding='utf-8').write(src)
print('packer/index.html', len(src))
