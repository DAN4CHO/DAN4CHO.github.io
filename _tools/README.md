# Инструменты 3D-просмотрщика (Jekyll не публикует папки с `_`)

- `optimize_glb.py` — сжатие GLB: `python3 -I _tools/optimize_glb.py in.glb out.glb 2`
- `pack_viewer.py` — куски модели + index.html: `python3 -I _tools/pack_viewer.py out.glb _tools/viewer_template.html <slug> "<Название>"`
- `viewer_template.html` — шаблон страницы (срез по высоте, прогулка от первого лица, фикс цветов SketchUp, NeutralToneMapping)
- `rerender.py` — пересобрать index.html проекта под новый шаблон, не трогая модель: `python3 -I _tools/rerender.py _tools/viewer_template.html <slug>/index.html <slug>/index.html`
- `core.js` + `packer.src.html` + `build_packer.py` — упаковщик в браузере (`packer/`), JS-порт optimize_glb.py; после правки шаблона или core.js: `python3 -I _tools/build_packer.py`
