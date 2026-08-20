import { FileCode2 } from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import { ShaderEditorActions } from './ShaderSourceEditor';

export function ShaderSourceEditorToolbar({ document }: { document: EditorDocument }) {
  return (
    <div className="main-toolbar shader-document-toolbar">
      <div className="toolbar-left">
        <span className="toolbar-group shader-document-toolbar-label">
          <FileCode2 size={15} />
          <span>Shader</span>
        </span>
        <span className="toolbar-separator" />
        <ShaderEditorActions document={document} />
      </div>
    </div>
  );
}
