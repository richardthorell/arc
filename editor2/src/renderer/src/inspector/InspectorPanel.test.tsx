// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { InspectorPanel } from './InspectorPanel';
import type { InspectorEntitySnapshot } from './inspectorTypes';

const cameraSnapshot = (): InspectorEntitySnapshot => ({
  entity: { index: 3, generation: 1 },
  name: 'Main Camera',
  tag: 'Camera',
  active: true,
  renderLayerMask: 1,
  transform: {
    position: { x: 1, y: 2, z: 3 },
    rotationDegrees: { x: 0, y: -18, z: 0 },
    rotationQuaternion: { x: 0, y: 0, z: 0, w: 1 },
    scale: { x: 1, y: 1, z: 1 },
  },
  camera: {
    projection: 'perspective',
    fovYDegrees: 60,
    orthographicHeight: 10,
    nearPlane: 0.1,
    farPlane: 2000,
    active: true,
    clearColor: { x: 0.055, y: 0.12, z: 0.22, w: 1 },
  },
  meshRenderer: null,
  terrain: null,
  components: [
    { kind: 'transform', label: 'Transform', editable: true },
    { kind: 'camera', label: 'Camera', editable: true },
  ],
});

const meshSnapshot = (): InspectorEntitySnapshot => ({
  ...cameraSnapshot(),
  name: 'Mountain',
  tag: 'Mesh',
  camera: null,
  meshRenderer: {
    visible: true,
    baseColorTint: { x: 1, y: 1, z: 1, w: 1 },
    hasMaterial: true,
    assetBackedMaterial: true,
    materialName: 'Mountain Landscape',
    materialPath: 'materials/mountain_landscape.arcmat',
  },
  components: [
    { kind: 'transform', label: 'Transform', editable: true },
    { kind: 'meshRenderer', label: 'Mesh Renderer', editable: true },
  ],
});

const terrainSnapshot = (): InspectorEntitySnapshot => ({
  ...cameraSnapshot(),
  name: 'Terrain',
  tag: 'Environment',
  camera: null,
  terrain: {
    enabled: true,
    size: 180,
    resolution: 257,
    chunkQuads: 128,
    receiveShadows: true,
    contentRevision: 3,
    brushTool: 'sculpt',
    brushRadius: 6,
    brushStrength: 0.25,
    brushFalloff: 1,
    activeLayer: 0,
    layers: [
      { name: 'Grass', baseColorPath: 'textures/grass.jpg' },
      { name: 'Dirt', baseColorPath: '' },
      { name: 'Rock', baseColorPath: '' },
      { name: 'Sand', baseColorPath: '' },
    ],
  },
  components: [
    { kind: 'transform', label: 'Transform', editable: true },
    { kind: 'terrain', label: 'Terrain', editable: true },
  ],
});

describe('data-driven InspectorPanel', () => {
  beforeEach(() => {
    window.requestAnimationFrame = (callback) => window.setTimeout(() => callback(performance.now()), 0);
    window.cancelAnimationFrame = (handle) => window.clearTimeout(handle);
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it('renders schemas and switches projection-dependent camera fields', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true });
    render(<InspectorPanel snapshot={cameraSnapshot()} command={command} refresh={async () => undefined} />);

    expect(screen.getByText('Transform')).toBeInTheDocument();
    expect(screen.getByLabelText('Collapse Camera')).toBeInTheDocument();
    expect(screen.getByLabelText('Field of View')).toBeInTheDocument();
    expect(screen.queryByLabelText('Ortho Size')).not.toBeInTheDocument();

    await userEvent.selectOptions(screen.getByLabelText('Projection'), 'orthographic');
    expect(screen.getByLabelText('Ortho Size')).toBeInTheDocument();
    expect(screen.queryByLabelText('Field of View')).not.toBeInTheDocument();
    expect(command).toHaveBeenCalledWith('entity.setCamera', expect.objectContaining({
      entity: { index: 3, generation: 1 },
      camera: expect.objectContaining({ projection: 'orthographic' }),
    }));
  });

  it('renders terrain and virtual brush schemas and sends typed brush and layer commands', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true });
    const assets = [{ id: 'sand', name: 'Sand', path: 'textures/sand.jpg', kind: 'texture', status: 'ready' as const }];
    render(<InspectorPanel snapshot={terrainSnapshot()} command={command} refresh={async () => undefined} assets={assets} />);

    expect(screen.getByLabelText('Collapse Terrain')).toBeInTheDocument();
    expect(screen.getByLabelText('Collapse Terrain Brush')).toBeInTheDocument();
    expect(screen.getByLabelText('Choose Grass Layer asset')).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText('Tool'), 'paint');
    await waitFor(() => expect(command).toHaveBeenCalledWith('terrain.setBrush', expect.objectContaining({ tool: 'paint' })));
    expect(screen.getByLabelText('Paint Layer')).toBeInTheDocument();

    await userEvent.click(screen.getByLabelText('Choose Sand Layer asset'));
    await userEvent.click(screen.getByLabelText('Select Sand'));
    await waitFor(() => expect(command).toHaveBeenCalledWith('terrain.assignLayer', expect.objectContaining({ layer: 3, path: 'textures/sand.jpg' })));
  });

  it('filters, collapses, and restores component content', async () => {
    render(<InspectorPanel snapshot={cameraSnapshot()} command={vi.fn().mockResolvedValue({ succeeded: true })} refresh={async () => undefined} />);
    await userEvent.type(screen.getByLabelText('Search components'), 'camera');
    expect(screen.queryByText('Transform')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Collapse Camera')).toBeInTheDocument();

    await userEvent.clear(screen.getByLabelText('Search components'));
    await userEvent.click(screen.getByLabelText('Collapse Transform'));
    expect(screen.queryByLabelText('Location X')).not.toBeInTheDocument();
    await userEvent.click(screen.getByLabelText('Expand Transform'));
    expect(screen.getByLabelText('Location X')).toBeInTheDocument();
  });

  it('commits header metadata through typed commands', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true });
    render(<InspectorPanel snapshot={cameraSnapshot()} command={command} refresh={async () => undefined} />);

    await userEvent.click(screen.getByLabelText('Entity active'));
    const tag = screen.getByLabelText('Tag');
    await userEvent.clear(tag);
    await userEvent.type(tag, 'Environment{Enter}');
    await userEvent.selectOptions(screen.getByLabelText('Layer'), '2');
    const name = screen.getByLabelText('Entity name');
    await userEvent.clear(name);
    await userEvent.type(name, 'Gameplay Camera{Enter}');

    await waitFor(() => expect(command).toHaveBeenCalledTimes(4));
    expect(command).toHaveBeenCalledWith('entity.setActive', expect.objectContaining({ active: false }));
    expect(command).toHaveBeenCalledWith('entity.setTag', expect.objectContaining({ tag: 'Environment' }));
    expect(command).toHaveBeenCalledWith('entity.setRenderLayer', expect.objectContaining({ renderLayerMask: 2 }));
    expect(command).toHaveBeenCalledWith('entity.rename', expect.objectContaining({ name: 'Gameplay Camera' }));
    expect(screen.getByLabelText('Static')).toBeDisabled();
  });

  it('commits typed numbers, supports arrow steps, and rolls back rejected edits', async () => {
    const command = vi.fn()
      .mockResolvedValueOnce({ succeeded: true })
      .mockResolvedValueOnce({ succeeded: false, error: 'Rejected by host' });
    render(<InspectorPanel snapshot={cameraSnapshot()} command={command} refresh={async () => undefined} />);
    const locationX = screen.getByLabelText('Location X');
    fireEvent.change(locationX, { target: { value: '12.5' } });
    fireEvent.keyDown(locationX, { key: 'Enter' });
    await waitFor(() => expect(command).toHaveBeenCalledTimes(1));
    expect(command).toHaveBeenLastCalledWith('entity.setTransform', expect.objectContaining({
      transform: expect.objectContaining({ position: [12.5, 2, 3] }),
    }));

    const near = screen.getByLabelText('Near Clip');
    fireEvent.keyDown(near, { key: 'ArrowUp' });
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Rejected by host'));
    expect(screen.getByLabelText('Near Clip')).toHaveValue('0.100');
  });

  it('links scale axes proportionally', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true });
    render(<InspectorPanel snapshot={cameraSnapshot()} command={command} refresh={async () => undefined} />);
    await userEvent.click(screen.getByLabelText('Link scale axes'));
    const scaleX = screen.getByLabelText('Scale X');
    fireEvent.change(scaleX, { target: { value: '2' } });
    fireEvent.keyDown(scaleX, { key: 'Enter' });
    await waitFor(() => expect(command).toHaveBeenCalled());
    expect(command).toHaveBeenLastCalledWith('entity.setTransform', expect.objectContaining({
      transform: expect.objectContaining({ scale: [2, 2, 2] }),
    }));
  });

  it('throttles drag scrubbing and sends a final transform value', async () => {
    vi.useFakeTimers();
    const command = vi.fn().mockResolvedValue({ succeeded: true });
    render(<InspectorPanel snapshot={cameraSnapshot()} command={command} refresh={async () => undefined} />);
    const scrubber = document.querySelector('.inspector-number-scrub.axis-x');
    expect(scrubber).not.toBeNull();
    fireEvent.pointerDown(scrubber!, { button: 0, clientX: 0 });
    fireEvent.pointerMove(window, { clientX: 10 });
    fireEvent.pointerMove(window, { clientX: 20 });
    await vi.runAllTimersAsync();
    fireEvent.pointerUp(window);
    await vi.runAllTimersAsync();
    expect(command).toHaveBeenCalledTimes(2);
    expect(command).toHaveBeenLastCalledWith('entity.setTransform', expect.objectContaining({
      transform: expect.objectContaining({ position: [1.4, 2, 3] }),
    }), expect.objectContaining({ phase: 'commit', label: 'Transform Entity' }));
    vi.useRealTimers();
  });

  it('ignores stale rejected responses after a newer successful edit', async () => {
    let rejectOld: ((value: { succeeded: boolean; error: string }) => void) | undefined;
    const oldResponse = new Promise<{ succeeded: boolean; error: string }>((resolve) => { rejectOld = resolve; });
    const command = vi.fn()
      .mockReturnValueOnce(oldResponse)
      .mockResolvedValueOnce({ succeeded: true });
    render(<InspectorPanel snapshot={cameraSnapshot()} command={command} refresh={async () => undefined} />);
    fireEvent.click(screen.getByLabelText('Entity active'));
    fireEvent.change(screen.getByLabelText('Layer'), { target: { value: '2' } });
    await waitFor(() => expect(command).toHaveBeenCalledTimes(2));
    rejectOld?.({ succeeded: false, error: 'Stale failure' });
    await Promise.resolve();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Layer')).toHaveValue('2');
  });

  it('opens the advanced color picker and commits sRGB hex as linear RGBA', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true });
    render(<InspectorPanel snapshot={cameraSnapshot()} command={command} refresh={async () => undefined} />);

    await userEvent.click(screen.getByLabelText('Open Clear Color color picker'));
    expect(screen.getByRole('dialog', { name: 'Clear Color color picker' })).toBeInTheDocument();
    expect(screen.getByLabelText('Saturation and value')).toBeInTheDocument();
    expect(screen.getByLabelText('Hue')).toBeInTheDocument();
    expect(screen.getByLabelText('Alpha')).toBeInTheDocument();
    expect(screen.getByLabelText('Restore original Clear Color')).toBeInTheDocument();

    const hex = screen.getByLabelText('Hex sRGB');
    await userEvent.clear(hex);
    await userEvent.type(hex, '#FF000080{Enter}');
    await waitFor(() => expect(command).toHaveBeenCalledWith('entity.setCamera', expect.objectContaining({
      camera: expect.objectContaining({ clearColor: expect.arrayContaining([1, 0, 0]) }),
    })));
    const clearColor = command.mock.calls.at(-1)?.[1]?.camera?.clearColor as number[];
    expect(clearColor[3]).toBeCloseTo(128 / 255, 6);

    await userEvent.click(screen.getByRole('button', { name: 'HSV' }));
    expect(screen.getByLabelText('Color H')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Linear' })).toBeDisabled();
  });

  it('renders the Mesh Renderer material preview and assigns a material asset', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true });
    const thumbnailProvider = vi.fn().mockResolvedValue('data:image/bmp;base64,Qkpreview');
    render(<InspectorPanel snapshot={meshSnapshot()} command={command} refresh={async () => undefined}
      assets={[
        { id: 'mountain', name: 'Mountain Landscape', path: 'materials/mountain_landscape.arcmat', kind: 'material', status: 'ready' },
        { id: 'bronze', name: 'Bronze', path: 'materials/bronze.arcmat', kind: 'material', status: 'ready' },
        { id: 'rock', name: 'Rock Albedo', path: 'textures/rock.png', kind: 'texture', status: 'ready' },
      ]}
      thumbnailProvider={thumbnailProvider} />);

    expect(screen.getByLabelText('Material Preview')).toBeInTheDocument();
    expect(screen.getAllByText('Mountain Landscape')).toHaveLength(2);
    await userEvent.click(screen.getByLabelText('Choose Material asset'));
    expect(screen.getByRole('dialog', { name: 'Material asset picker' })).toBeInTheDocument();
    expect(screen.queryByLabelText('Select Rock Albedo')).not.toBeInTheDocument();
    await userEvent.click(screen.getByLabelText('Select Bronze'));
    await waitFor(() => expect(command).toHaveBeenCalledWith('entity.setMaterial', {
      entity: { index: 3, generation: 1 }, path: 'materials/bronze.arcmat',
    }));

    await userEvent.click(screen.getByLabelText('Visible'));
    await waitFor(() => expect(command).toHaveBeenCalledWith('entity.setMeshRenderer', expect.objectContaining({
      visible: false, baseColorTint: [1, 1, 1, 1],
    })));
  });
});
