import { FileCode2 } from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import type { AssetItem } from '../services/editorHostTypes';
import { ShaderSourceEditor } from './ShaderSourceEditor';

export function ShaderEditorPanel({ asset }: { asset: AssetItem | null }) {
  const shader = asset?.kind === 'shader' ? asset : null;

  if (!shader)
    return (
      <section className="production-tool-panel tool-empty-state">
        <FileCode2 size={29} />
        <strong>Select a shader asset</strong>
        <span>The shader editor only opens registered project shader sources.</span>
      </section>
    );

  const document: EditorDocument = {
    id: `legacy-shader:${shader.guid ?? shader.id ?? shader.path}`,
    kind: 'shader',
    title: shader.name,
    path: shader.path,
    assetId: shader.id,
    assetGuid: shader.guid,
    assetScope: shader.scope,
    dirty: false,
    readOnly: shader.scope === 'builtin' || Boolean(shader.readOnly),
  };

  return <ShaderSourceEditor document={document} embeddedToolbar />;
}
