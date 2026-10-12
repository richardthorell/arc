// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MaterialParameterSubsection } from './MaterialParameterSubsection';

const entityGuid = '11111111-1111-4111-8111-111111111111';
const materialGuid = '22222222-2222-4222-8222-222222222222';
const parameterId = '18446744073709551615';
const query = vi.fn();
const command = vi.fn();
const readText = vi.fn();
let onEvent: (event: { type: string }) => void;
const response = (payload: unknown) => ({ succeeded: true, payload, sceneRevision: 8, worldEpoch: 3 });
const description = () => ({
  guid: materialGuid,
  kind: 'material',
  provenance: 'savedSource',
  definitionRevision: 'a'.repeat(64),
  parentGuid: null,
  dependencies: [],
  functionSlots: [],
  functionInterface: null,
  diagnostics: [],
  definitionReady: true,
  runtimeStatus: 'notInspected',
  entity: { guid: entityGuid, materialGuid, runtimeHandle: { index: 5, generation: 2 }, resourceAlive: true },
  parameters: [
    {
      id: parameterId,
      name: 'Artist Control',
      type: 'float',
      defaultValue: [0.25],
      inheritedValue: [0.25],
      resolvedValue: [0.5],
      overridden: true,
      slotId: '',
      semantic: 'none',
      range: { min: 0, max: 1 },
      valueStatus: 'resolved',
    },
  ],
});
beforeEach(() => {
  vi.clearAllMocks();
  query.mockImplementation(async (type) =>
    response(type === 'entity.selected' ? { guid: entityGuid, selectionCount: 1 } : description()),
  );
  command.mockResolvedValue(response({ applied: true }));
  Object.defineProperty(window, 'arc', {
    configurable: true,
    value: {
      projects: { readText },
      host: {
        query,
        command,
        onEvent: (callback: typeof onEvent) => {
          onEvent = callback;
          return () => {};
        },
      },
    },
  });
});
afterEach(cleanup);
const show = () => render(<MaterialParameterSubsection assets={[]} value={materialGuid} />);

describe('native-backed entity material controls', () => {
  it('renders dynamic compiler metadata without reading or evaluating a graph document', async () => {
    show();
    expect(await screen.findByLabelText('Artist Control')).toHaveValue('0.500');
    expect(query).toHaveBeenCalledWith('materials.describe', { entityGuid });
    expect(readText).not.toHaveBeenCalled();
    expect(command).not.toHaveBeenCalled();
  });
  it('does not present internal one-choice Material Function slots as editable selections', async () => {
    query.mockImplementation(async (type) =>
      response(
        type === 'entity.selected'
          ? { guid: entityGuid, selectionCount: 1 }
          : {
              ...description(),
              functionSlots: [
                { id: 'internal', name: 'Unpack ORM', selectedFunctionGuid: 'a', alternatives: [{ guid: 'a', pathHint: 'orm' }] },
                {
                  id: 'authorable',
                  name: 'Base Color Source',
                  selectedFunctionGuid: 'a',
                  alternatives: [
                    { guid: 'a', pathHint: 'color' },
                    { guid: 'b', pathHint: 'checker' },
                  ],
                },
              ],
            },
      ),
    );
    show();
    expect(await screen.findByLabelText('Base Color Source function')).toBeInTheDocument();
    expect(screen.queryByLabelText('Unpack ORM function')).not.toBeInTheDocument();
  });
  it('nests exposed function parameters beneath their selectable function and retains range sliders', async () => {
    query.mockImplementation(async (type) =>
      response(
        type === 'entity.selected'
          ? { guid: entityGuid, selectionCount: 1 }
          : {
              ...description(),
              parameters: [
                { ...description().parameters[0], slotId: 'base-color-source', name: 'Color Strength' },
              ],
              functionSlots: [
                {
                  id: 'base-color-source',
                  name: 'Base Color Source',
                  defaultFunctionGuid: '33333333-3333-4333-8333-333333333333',
                  selectedFunctionGuid: '33333333-3333-4333-8333-333333333333',
                  inputs: [],
                  outputs: [],
                  alternatives: [
                    { guid: '33333333-3333-4333-8333-333333333333', pathHint: 'color' },
                    { guid: '44444444-4444-4444-8444-444444444444', pathHint: 'checker' },
                  ],
                  alternativesTruncated: false,
                },
              ],
            },
      ),
    );
    const { container } = show();
    expect(await screen.findByLabelText('Base Color Source function')).toBeInTheDocument();
    const nested = container.querySelector('.inspector-material-function-parameters');
    expect(nested).not.toBeNull();
    expect(nested).toContainElement(screen.getByLabelText('Color Strength'));
    expect(nested).toContainElement(screen.getByLabelText('Color Strength slider'));
  });
  it('commits a typed 64-bit override with inspected scene, world and definition authority', async () => {
    show();
    const input = await screen.findByLabelText('Artist Control');
    fireEvent.change(input, { target: { value: '0.75' } });
    fireEvent.blur(input);
    await waitFor(() =>
      expect(command).toHaveBeenCalledWith('materials.edit', {
        operation: 'setOverride',
        entityGuid,
        parameterId,
        value: [0.75],
        definitionRevision: 'a'.repeat(64),
        expectedSceneRevision: 8,
        worldEpoch: 3,
      }),
    );
    await waitFor(() => expect(query.mock.calls.filter(([type]) => type === 'materials.describe')).toHaveLength(2));
    expect(readText).not.toHaveBeenCalled();
  });
  it('resets only the selected entity parameter through the same typed service', async () => {
    show();
    fireEvent.click(await screen.findByLabelText('Reset Artist Control'));
    await waitFor(() =>
      expect(command).toHaveBeenCalledWith(
        'materials.edit',
        expect.objectContaining({
          operation: 'resetOverride',
          parameterId,
          entityGuid,
        }),
      ),
    );
  });
  it('reports a rejected generation and refreshes last-good authority without claiming success', async () => {
    command.mockResolvedValue({ succeeded: false, error: 'Material definition revision is stale' });
    show();
    fireEvent.click(await screen.findByLabelText('Reset Artist Control'));
    expect(await screen.findByRole('alert')).toHaveTextContent('Material definition revision is stale');
    await waitFor(() => expect(query.mock.calls.filter(([type]) => type === 'materials.describe')).toHaveLength(2));
    expect(screen.getByLabelText('Artist Control')).toHaveValue('0.500');
  });
  it('refreshes resolved values on undo or external asset changes', async () => {
    show();
    await screen.findByLabelText('Artist Control');
    onEvent({ type: 'scene.changed' });
    await waitFor(() => expect(query.mock.calls.filter(([type]) => type === 'materials.describe')).toHaveLength(2));
  });
  it('does not edit a hidden multi-selection or route procedural controls through material APIs', async () => {
    query.mockResolvedValue(response({ guid: entityGuid, selectionCount: 2 }));
    const view = show();
    expect(await screen.findByRole('alert')).toHaveTextContent('Select one entity');
    expect(command).not.toHaveBeenCalled();
    query.mockClear();
    view.rerender(<MaterialParameterSubsection assets={[]} value="__arc_primitive_parameter__/width/2" />);
    expect(query).not.toHaveBeenCalled();
  });
});
