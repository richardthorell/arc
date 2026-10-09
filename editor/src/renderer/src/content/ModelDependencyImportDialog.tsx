import { FileBox, FileImage, FileText } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import type { ExternalModelImportPlan } from '../../../preload/externalModelImport';
import { UiButton, UiDialog } from '../ui';

import './modelDependencyImportDialog.css';

type Props = {
  plan: ExternalModelImportPlan;
  destination: string;
  onCancel: () => void;
  onImport: (selectedDependencies: string[]) => void;
};

const DependencyIcon = ({ kind }: { kind: ExternalModelImportPlan['dependencies'][number]['kind'] }) =>
  kind === 'texture' ? <FileImage aria-hidden="true" size={17} /> : kind === 'material' ? <FileText aria-hidden="true" size={17} /> : <FileBox aria-hidden="true" size={17} />;

export function ModelDependencyImportDialog({ plan, destination, onCancel, onImport }: Props) {
  const availablePaths = useMemo(
    () => plan.dependencies.filter((dependency) => dependency.exists).map((dependency) => dependency.path),
    [plan],
  );
  const [selected, setSelected] = useState<Set<string>>(() => new Set(availablePaths));

  useEffect(() => {
    setSelected(new Set(availablePaths));
  }, [availablePaths, plan.sourcePath]);

  const toggle = (path: string) =>
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });

  return (
    <UiDialog
      className="model-dependency-import-dialog"
      footer={
        <>
          <UiButton onClick={onCancel}>Cancel</UiButton>
          <UiButton
            variant="primary"
            onClick={() => onImport(plan.dependencies.filter((item) => selected.has(item.path)).map((item) => item.path))}
          >
            Import
          </UiButton>
        </>
      }
      icon={<FileBox aria-hidden="true" size={18} />}
      onClose={onCancel}
      subtitle={`Choose which referenced assets to copy into ${destination}`}
      title={`Import ${plan.fileName}`}
      width={640}
    >
      <div className="model-dependency-import-toolbar">
        <div>
          <strong>Referenced assets</strong>
          <span>
            {selected.size} of {availablePaths.length} selected
          </span>
        </div>
        <div>
          <UiButton
            disabled={selected.size === availablePaths.length}
            onClick={() => setSelected(new Set(availablePaths))}
          >
            Select All
          </UiButton>
          <UiButton disabled={selected.size === 0} onClick={() => setSelected(new Set())}>
            Clear
          </UiButton>
        </div>
      </div>

      <div className="model-dependency-import-list">
        {plan.dependencies.map((dependency) => (
          <label
            className={`model-dependency-import-row ${dependency.exists ? '' : 'missing'}`}
            key={dependency.path}
          >
            <input
              aria-label={dependency.path}
              checked={dependency.exists && selected.has(dependency.path)}
              disabled={!dependency.exists}
              type="checkbox"
              onChange={() => toggle(dependency.path)}
            />
            <DependencyIcon kind={dependency.kind} />
            <span>
              <strong>{dependency.path}</strong>
              <small>
                {dependency.kind === 'material'
                  ? 'Material'
                  : dependency.kind === 'texture'
                    ? 'Texture'
                    : dependency.kind === 'buffer'
                      ? 'Buffer'
                      : 'Referenced file'}
                {!dependency.exists ? ' · Missing' : ''}
              </small>
            </span>
          </label>
        ))}
      </div>
    </UiDialog>
  );
}
