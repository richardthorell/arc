import { useEffect, useRef, useState } from 'react';
import { ChevronDown, Eye, RotateCcw, Save } from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import { UiButton, UiFloatingSurface, UiSelect, UiSlider, UiSplitButton } from '../ui';
import {
  hasPendingTextureSettings,
  revertTextureDocument,
  saveTextureDocument,
  useTextureDocumentState,
} from './textureDocumentState';
import {
  getTextureEditorViewState,
  setTextureEditorViewState,
  updateTextureChannels,
  useTextureEditorViewState,
} from './textureEditorViewState';

import '../layout/toolbar.css';
import './textureEditorToolbar.css';

const previewModes = ['source', 'processed', 'difference'] as const;

export function TextureEditorToolbar({ document }: { document: EditorDocument }) {
  const state = useTextureEditorViewState(document.id);
  const documentState = useTextureDocumentState(document.id);
  const [viewOpen, setViewOpen] = useState(false);
  const viewRootRef = useRef<HTMLSpanElement | null>(null);
  const mipLevels = Math.max(1, document.assetSnapshot?.mipLevels ?? 1);
  const maxMip = mipLevels - 1;
  const mipOptions = Array.from({ length: mipLevels }, (_, level) => ({
    value: String(level),
    label: `Mip Level ${level}`,
  }));
  const hasPendingChanges = hasPendingTextureSettings(documentState);
  const setMipLevel = (value: number) =>
    setTextureEditorViewState(document.id, { mipLevel: Math.max(0, Math.min(maxMip, value)) });
  const toggleChannel = (channel: keyof typeof state.channels) =>
    updateTextureChannels(document.id, {
      [channel]: !getTextureEditorViewState(document.id).channels[channel],
    });

  useEffect(() => {
    if (!viewOpen) return;
    const close = (event: PointerEvent) => {
      if (!viewRootRef.current?.contains(event.target as Node)) setViewOpen(false);
    };
    window.addEventListener('pointerdown', close);
    return () => window.removeEventListener('pointerdown', close);
  }, [viewOpen]);

  return (
    <div aria-label="Texture editor toolbar" className="main-toolbar texture-document-toolbar">
      <div className="toolbar-left">
        <UiSplitButton
          ariaLabel={documentState.saving ? 'Saving texture' : 'Save texture'}
          className="texture-toolbar-save"
          disabled={document.readOnly || documentState.saving || !hasPendingChanges}
          icon={<Save size={13} />}
          label={documentState.saving ? 'Saving…' : 'Save'}
          menuAriaLabel="Texture save actions"
          onClick={() => void saveTextureDocument(document)}
          onOptionSelect={(value) => {
            if (value === 'revert') void revertTextureDocument(document);
          }}
          options={[
            {
              value: 'revert',
              label: 'Revert Unsaved Changes',
              icon: <RotateCcw size={13} />,
              disabled: documentState.saving || !hasPendingChanges,
            },
          ]}
          variant="toolbar"
        />
      </div>

      <div className="toolbar-center">
        <div aria-label="Texture preview mode" className="ui-toolbar-group toolbar-group texture-toolbar-preview-modes">
          {previewModes.map((mode) => (
            <UiButton
              active={state.previewMode === mode}
              aria-pressed={state.previewMode === mode}
              key={mode}
              onClick={() => setTextureEditorViewState(document.id, { previewMode: mode })}
              type="button"
              variant="toolbar"
            >
              {mode[0].toUpperCase() + mode.slice(1)}
            </UiButton>
          ))}
        </div>

        <span aria-hidden="true" className="toolbar-separator" />

        <div aria-label="Texture channels" className="ui-toolbar-group toolbar-group texture-channel-group">
          {(['r', 'g', 'b', 'a'] as const).map((channel) => (
            <UiButton
              active={state.channels[channel]}
              aria-pressed={state.channels[channel]}
              className={`texture-channel texture-channel-${channel}`}
              key={channel}
              onClick={() => toggleChannel(channel)}
              type="button"
              variant="toolbar"
            >
              {channel.toUpperCase()}
            </UiButton>
          ))}
        </div>

        <span aria-hidden="true" className="toolbar-separator" />

        <div aria-label="Texture mip level" className="ui-toolbar-group toolbar-group texture-mip-group">
          <UiButton
            aria-label="Decrease mip level"
            className="texture-mip-step"
            disabled={state.mipLevel <= 0}
            onClick={() => setMipLevel(state.mipLevel - 1)}
            type="button"
            variant="toolbar"
          >
            −
          </UiButton>
          <UiSelect
            ariaLabel="Mip level"
            className="texture-mip-select"
            disabled={mipLevels <= 1}
            options={mipOptions}
            value={String(state.mipLevel)}
            onValueChange={(value) => setMipLevel(Number(value))}
          />
          <UiButton
            aria-label="Increase mip level"
            className="texture-mip-step"
            disabled={state.mipLevel >= maxMip}
            onClick={() => setMipLevel(state.mipLevel + 1)}
            type="button"
            variant="toolbar"
          >
            +
          </UiButton>
        </div>
      </div>

      <div className="toolbar-right">
        <span className="texture-toolbar-view-menu" ref={viewRootRef}>
          <UiButton
            aria-expanded={viewOpen}
            aria-haspopup="dialog"
            onClick={() => setViewOpen((open) => !open)}
            type="button"
            variant="toolbar"
          >
            <Eye size={13} /> View <ChevronDown aria-hidden="true" size={12} />
          </UiButton>
          {viewOpen && (
            <UiFloatingSurface
              aria-label="Texture preview view options"
              className="texture-toolbar-view-popup"
              role="dialog"
              width={252}
            >
              <div className="texture-toolbar-view-row">
                <span>Preview Filter</span>
                <UiSelect
                  ariaLabel="Texture preview sampling"
                  className="texture-view-sampling-select"
                  options={[
                    { value: 'linear', label: 'Linear' },
                    { value: 'nearest', label: 'Nearest' },
                  ]}
                  value={state.sampling}
                  onValueChange={(value) =>
                    setTextureEditorViewState(document.id, {
                      sampling: value === 'nearest' ? 'nearest' : 'linear',
                    })
                  }
                />
              </div>
              <div className="texture-toolbar-view-row texture-toolbar-exposure-row">
                <span>Exposure</span>
                <UiSlider
                  aria-label="Texture preview exposure"
                  max={8}
                  min={-8}
                  onValueChange={(exposure) => setTextureEditorViewState(document.id, { exposure })}
                  step={0.25}
                  value={state.exposure}
                />
                <output>
                  {state.exposure > 0 ? '+' : ''}
                  {state.exposure.toFixed(2)} EV
                </output>
              </div>
            </UiFloatingSurface>
          )}
        </span>
      </div>
    </div>
  );
}
