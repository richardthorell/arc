from pathlib import Path


def rep(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    if old not in text:
        raise RuntimeError(f"missing pattern in {path}: {old[:100]}")
    file.write_text(text.replace(old, new, 1))


p = "editor/src/renderer/src/viewport/ViewportPanel.tsx"
rep(
    p,
    "import { assignDroppedMaterialToViewport } from './viewportAssetDrop';",
    "import { assignDroppedMaterialToViewport, instantiateDroppedMeshInViewport } from './viewportAssetDrop';",
)
rep(
    p,
    """  const onAssetDragOver = (event: DragEvent<HTMLDivElement>) => {
    if (!viewportActive) return;
    if (
      !event.dataTransfer.types.includes(arcAssetDragMime) &&
      !event.dataTransfer.types.includes(arcEnvironmentDragMime)
    )
      return;
""",
    """  const onAssetDragOver = (event: DragEvent<HTMLDivElement>) => {
    if (!viewportActive) return;
    const externalModel =
      event.dataTransfer.types.includes('Files') &&
      Array.from(event.dataTransfer.files).some((file) => /\.(fbx|glb|gltf|obj)$/i.test(file.name));
    if (
      !externalModel &&
      !event.dataTransfer.types.includes(arcAssetDragMime) &&
      !event.dataTransfer.types.includes(arcEnvironmentDragMime)
    )
      return;
""",
)
rep(
    p,
    """    const dropped = readArcAssetDragPayload(event.dataTransfer);
    if (!dropped || dropped.type !== 'material') return;
    event.preventDefault();
    event.stopPropagation();
    const position = pointerCoordinates(event.clientX, event.clientY);
    void assignDroppedMaterialToViewport(window.arc.host, {
      viewportId,
      ...position,
      path: dropped.pathHint,
    }).then((result) => {
      if (result.succeeded) {
        setViewportError('');
      } else if (result.error !== 'No scene object at the drop position') {
        setViewportError(result.error);
      }
    });
""",
    """    const dropped = readArcAssetDragPayload(event.dataTransfer);
    if (!dropped) return;
    const position = pointerCoordinates(event.clientX, event.clientY);
    if (dropped.type === 'material') {
      event.preventDefault();
      event.stopPropagation();
      void assignDroppedMaterialToViewport(window.arc.host, {
        viewportId,
        ...position,
        path: dropped.pathHint,
      }).then((result) => {
        if (result.succeeded) setViewportError('');
        else if (result.error !== 'No scene object at the drop position') setViewportError(result.error);
      });
      return;
    }
    if (dropped.type !== 'mesh' && dropped.type !== 'scene') return;
    event.preventDefault();
    event.stopPropagation();
    void instantiateDroppedMeshInViewport(window.arc.host, {
      viewportId,
      ...position,
      path: dropped.pathHint,
    }).then((result) => {
      if (result.succeeded) setViewportError('');
      else setViewportError(result.error);
    });
""",
)

asset_drop = Path("editor/src/renderer/src/viewport/viewportAssetDrop.ts")
asset_drop.write_text(
    asset_drop.read_text()
    + r'''

export type ViewportMeshDropResult = { succeeded: true; entity: HostEntityId } | { succeeded: false; error: string };

const parseEntity = (payload: unknown): HostEntityId | undefined => {
  if (!payload || typeof payload !== 'object') return undefined;
  const entity = (payload as { entity?: HostEntityId }).entity;
  return validEntity(entity) ? entity : undefined;
};

const parseWorldPosition = (payload: unknown): [number, number, number] | undefined => {
  if (!payload || typeof payload !== 'object') return undefined;
  const value = (payload as { worldPosition?: unknown }).worldPosition;
  if (!Array.isArray(value) || value.length !== 3 || !value.every((entry) => typeof entry === 'number')) return undefined;
  return value as [number, number, number];
};

export async function instantiateDroppedMeshInViewport(
  host: HostBridge,
  request: { viewportId: string; x: number; y: number; path: string },
): Promise<ViewportMeshDropResult> {
  const pick = (await host.command('viewport.pick', {
    viewportId: request.viewportId,
    x: request.x,
    y: request.y,
  })) as HostResponse;
  if (pick?.succeeded === false) return { succeeded: false, error: pick.error || 'Viewport placement failed' };

  const created = (await host.command('entity.create', { kind: 'empty' })) as HostResponse;
  if (created?.succeeded === false) return { succeeded: false, error: created.error || 'Could not create mesh entity' };
  const entity = parseEntity(created?.payload);
  if (!entity) return { succeeded: false, error: 'Host did not return the created mesh entity' };

  const assignment = (await host.command('entity.setMaterial', {
    entity,
    path: `__arc_mesh__/${request.path}`,
  })) as HostResponse;
  if (assignment?.succeeded === false) {
    await host.command('entity.delete', { entity }).catch(() => undefined);
    return { succeeded: false, error: assignment.error || 'Could not assign the dropped mesh' };
  }

  const worldPosition = parseWorldPosition(pick?.payload);
  if (worldPosition) {
    const transformed = (await host.command('entity.setTransform', {
      entity,
      transform: { position: worldPosition, rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    })) as HostResponse;
    if (transformed?.succeeded === false)
      return { succeeded: false, error: transformed.error || 'Could not place the dropped mesh' };
  }

  const fileName = request.path.replaceAll('\\', '/').split('/').at(-1) ?? request.path;
  const name = fileName.replace(/\.[^.]+$/, '') || 'Mesh';
  const renamed = (await host.command('entity.rename', { entity, name })) as HostResponse;
  if (renamed?.succeeded === false) return { succeeded: false, error: renamed.error || 'Could not name the dropped mesh' };
  return { succeeded: true, entity };
}
'''
)

test = Path("editor/src/renderer/src/viewport/viewportAssetDrop.test.ts")
text = test.read_text().replace(
    "import { assignDroppedMaterialToViewport } from './viewportAssetDrop';",
    "import { assignDroppedMaterialToViewport, instantiateDroppedMeshInViewport } from './viewportAssetDrop';",
)
text += r'''

describe('instantiateDroppedMeshInViewport', () => {
  it('creates and places an existing mesh asset at the viewport drop point', async () => {
    const entity = { index: 8, generation: 2 };
    const command = vi
      .fn()
      .mockResolvedValueOnce({ succeeded: true, payload: { worldPosition: [1, 2, 3] } })
      .mockResolvedValueOnce({ succeeded: true, payload: { entity } })
      .mockResolvedValue({ succeeded: true });
    const result = await instantiateDroppedMeshInViewport(
      { command, query: vi.fn() },
      { viewportId: 'viewport-1', x: 10, y: 20, path: 'Content/Models/crate.obj' },
    );
    expect(result).toEqual({ succeeded: true, entity });
    expect(command).toHaveBeenNthCalledWith(1, 'viewport.pick', { viewportId: 'viewport-1', x: 10, y: 20 });
    expect(command).toHaveBeenNthCalledWith(2, 'entity.create', { kind: 'empty' });
    expect(command).toHaveBeenNthCalledWith(3, 'entity.setMaterial', {
      entity,
      path: '__arc_mesh__/Content/Models/crate.obj',
    });
    expect(command).toHaveBeenNthCalledWith(4, 'entity.setTransform', {
      entity,
      transform: { position: [1, 2, 3], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    });
    expect(command).toHaveBeenNthCalledWith(5, 'entity.rename', { entity, name: 'crate' });
  });
});
'''
test.write_text(text)

host = Path("editor/native/src/arc_host_base.inc")
text = host.read_text()
old = '''                editor_pick_result cpu_fallback{};
                if (camera && transform && viewport.valid())
                    cpu_fallback = pick_scene_entity(state_->scene.scene, *state_->renderer,
                                                     screen_ray_from_camera(*camera, *transform, viewport,
                                                                            static_cast<float>(payload.x),
                                                                            static_cast<float>(payload.y)));
'''
new = '''                editor_pick_result cpu_fallback{};
                editor_ray pick_ray{};
                math::vector3f drop_position{};
                if (camera && transform && viewport.valid())
                {
                    pick_ray = screen_ray_from_camera(*camera, *transform, viewport, static_cast<float>(payload.x),
                                                      static_cast<float>(payload.y));
                    cpu_fallback = pick_scene_entity(state_->scene.scene, *state_->renderer, pick_ray);
                    if (cpu_fallback.entity.valid() && std::isfinite(cpu_fallback.distance) && cpu_fallback.distance > 0.0f)
                        drop_position = pick_ray.origin + pick_ray.direction * cpu_fallback.distance;
                    else
                    {
                        const float denominator = pick_ray.direction[1];
                        const float ground_distance = std::abs(denominator) > 1.0e-5f ? -pick_ray.origin[1] / denominator : -1.0f;
                        drop_position = ground_distance > 0.0f ? pick_ray.origin + pick_ray.direction * ground_distance
                                                              : pick_ray.origin + pick_ray.direction * 10.0f;
                    }
                }
'''
if old not in text:
    raise RuntimeError("viewport pick CPU fallback pattern missing")
text = text.replace(old, new, 1)
pending = 'return success("{\\"pending\\":true}");'
if pending not in text:
    raise RuntimeError("pending pick response missing")
text = text.replace(
    pending,
    'return success("{\\"pending\\":true,\\"worldPosition\\":[" + std::to_string(drop_position[0]) + "," + std::to_string(drop_position[1]) + "," + std::to_string(drop_position[2]) + "]}");',
    1,
)
sync = 'return success("{\\"entity\\":" + to_json(to_host_entity(cpu_fallback.entity)) + \'}\');'
if sync not in text:
    raise RuntimeError("sync pick response missing")
text = text.replace(
    sync,
    'return success("{\\"entity\\":" + to_json(to_host_entity(cpu_fallback.entity)) + ",\\"worldPosition\\":[" + std::to_string(drop_position[0]) + "," + std::to_string(drop_position[1]) + "," + std::to_string(drop_position[2]) + "]}");',
    1,
)
host.write_text(text)
