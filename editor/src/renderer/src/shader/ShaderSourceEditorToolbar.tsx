import { FileCode2, Lock } from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import { ShaderEditorActions } from './ShaderSourceEditor';

export function ShaderSourceEditorToolbar({ document }: { document: EditorDocument }) {
  return (
    <div className="main-toolbar shader-document-toolbar">
      <div className="toolbar-left">
        <span className="toolbar-group shader-document-toolbar-label">
          <FileCode2 size={15} />
          <span>Shader</span>
          {document.readOnly && (
            <span title="Built-in engine asset">
              <Lock size={12} /> Read-only
            </span>
          )}
        </span>
        <span className="toolbar-separator" />
        <ShaderEditorActions document={document} />
      </div>
    </div>
  );
}
