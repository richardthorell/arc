from pathlib import Path

path = Path('editor/src/preload/assetSourceBridge.ts')
text = path.read_text()
old = """        stagingRoot = ensureContained(
          sourceRoot,
          path.join(sourceRoot, `.${normalizeSegment(request.assetId)}.import-${operationId}-${Date.now()}`),
        );
        await rm(stagingRoot, { recursive: true, force: true });
        await mkdir(stagingRoot, { recursive: true });
"""
new = """        const stagingParent = ensureContained(
          roots.savedRoot,
          path.join(roots.savedRoot, 'AssetImports', 'staging', normalizeSegment(request.sourceId)),
        );
        await mkdir(stagingParent, { recursive: true });
        stagingRoot = ensureContained(
          stagingParent,
          path.join(stagingParent, `.${normalizeSegment(request.assetId)}.import-${operationId}-${Date.now()}`),
        );
        await rm(stagingRoot, { recursive: true, force: true });
        await mkdir(stagingRoot, { recursive: true });
"""
if old not in text:
    raise SystemExit('staging anchor not found')
path.write_text(text.replace(old, new, 1))
