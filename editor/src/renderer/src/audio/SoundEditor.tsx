import { AudioLines, FileAudio2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import type { EditorDocument } from '../editors/editorTypes';
import { InspectorComponentCard } from '../inspector/InspectorComponentCard';
import type { AssetPickerItem } from '../inspector/AssetPicker';
import { soundPropertySchemas } from './soundSchemas';
import { setSoundDocumentValue, useSoundDocumentState } from './soundDocumentState';
import './soundEditor.css';

type HostProjectAssetsPayload = {
  assets?: Array<{
    guid?: string;
    path?: string;
    sourcePath?: string;
    title?: string;
    kind?: string;
    state?: AssetPickerItem['status'];
    scope?: AssetPickerItem['scope'];
    readOnly?: boolean;
  }>;
};

const basename = (path: string) => path.split(/[\\/]/).pop() || path;

export function SoundEditor({ document }: { document: EditorDocument }) {
  const state = useSoundDocumentState(document);
  const [assets, setAssets] = useState<AssetPickerItem[]>([]);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  useEffect(() => {
    let active = true;
    void window.arc.host
      .query('project.assets')
      .then((response: unknown) => {
        if (!active || !response || typeof response !== 'object') return;
        const payload = (response as { payload?: HostProjectAssetsPayload }).payload;
        const audio = (payload?.assets ?? [])
          .filter((asset) => asset.kind === 'audio' && Boolean(asset.path))
          .map<AssetPickerItem>((asset) => ({
            id: asset.guid || asset.path!,
            guid: asset.guid,
            name: asset.title?.trim() || basename(asset.path!),
            path: asset.path!,
            sourcePath: asset.sourcePath,
            kind: 'audio',
            status: asset.state ?? 'ready',
            scope: asset.scope,
            readOnly: asset.readOnly,
          }));
        setAssets(audio);
      })
      .catch(() => {
        if (active) setAssets([]);
      });
    return () => {
      active = false;
    };
  }, []);

  const sourceName = useMemo(
    () => (state.asset.source ? basename(state.asset.source) : 'No WAV source'),
    [state.asset.source],
  );

  return (
    <section className="sound-editor-workspace">
      <main className="sound-editor-work-area">
        <div className="sound-editor-stage">
          <div className="sound-editor-stage-icon" aria-hidden="true">
            <AudioLines size={42} />
          </div>
          <div className="sound-editor-stage-copy">
            <div className="sound-editor-stage-eyebrow">Sound</div>
            <h2>{document.title.replace(/\.arcsound$/i, '')}</h2>
            <div className="sound-editor-source-summary">
              <FileAudio2 aria-hidden="true" size={15} />
              <span>{sourceName}</span>
            </div>
          </div>
        </div>

        <div className="sound-editor-waveform-placeholder" aria-label="Sound work area">
          <div className="sound-editor-waveform-rule" />
          <div>
            <strong>Waveform workspace</strong>
            <span>Source analysis and playback controls will use this area when audio preview is added.</span>
          </div>
        </div>

        {state.message && (
          <div className="sound-editor-message" role="status">
            {state.message}
          </div>
        )}
      </main>

      <aside className="sound-editor-sidebar editor-property-panel" aria-label="Sound properties">
        <div className="sound-editor-sidebar-scroll">
          {soundPropertySchemas.map((schema) => (
            <InspectorComponentCard
              assets={assets}
              collapsed={collapsed[schema.id] ?? Boolean(schema.collapsedByDefault)}
              context={state.asset}
              key={schema.id}
              schema={schema}
              onToggle={() => setCollapsed((current) => ({ ...current, [schema.id]: !current[schema.id] }))}
              onValue={(path, value, settled) => setSoundDocumentValue(document, path, value, settled)}
            />
          ))}
        </div>
      </aside>
    </section>
  );
}
