#!/usr/bin/env python3
from pathlib import Path
terms = ['load_mesh', 'load_model', 'instantiate', 'mesh_result', 'skeletons', 'skin_index', 'SkinnedMeshRenderer', 'skinned_mesh_renderer_component']
roots = [Path('editor/native/src'), Path('engine/scene'), Path('engine/render')]
for root in roots:
    for path in root.rglob('*'):
        if not path.is_file() or path.suffix not in {'.cpp','.h','.inc','.hpp'}:
            continue
        try: text = path.read_text(errors='ignore')
        except: continue
        hits = [t for t in terms if t in text]
        if hits:
            print(path, ','.join(hits))
            lines=text.splitlines()
            for i,line in enumerate(lines):
                if any(t in line for t in hits):
                    lo=max(0,i-3); hi=min(len(lines),i+5)
                    print(f'--- {path}:{lo+1}-{hi}')
                    print('\n'.join(f'{j+1}: {lines[j]}' for j in range(lo,hi)))
