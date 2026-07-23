import { describe, expect, it } from 'vitest';

import { buildSceneTree, classifyHostEventRefresh, filterSceneTree } from './Workbench';
import type { HostSceneEntity } from './Workbench';

const entity = (index: number, guid: string, parentGuid = '', siblingOrder = 0, name = guid): HostSceneEntity => ({
  entity: { index, generation: 1 }, guid, parentGuid, siblingOrder, name,
  kind: 'mesh', active: true, selected: false,
});

describe('host scene hierarchy reconstruction', () => {
  it('reconstructs ordered parent relationships by stable GUID', () => {
    const tree = buildSceneTree([
      entity(3, 'child-b', 'parent', 1, 'Child B'),
      entity(1, 'parent', '', 0, 'Parent'),
      entity(4, 'root-b', '', 1, 'Root B'),
      entity(2, 'child-a', 'parent', 0, 'Child A'),
    ]);
    expect(tree.map((value) => value.name)).toEqual(['Parent', 'Root B']);
    expect(tree[0].children?.map((value) => value.name)).toEqual(['Child A', 'Child B']);
    expect(tree[0].children?.[0].parentId).toBe(tree[0].id);
    expect(tree[0].guid).toBe('parent');
  });

  it('retains ancestor paths while filtering descendants', () => {
    const tree = buildSceneTree([
      entity(1, 'world', '', 0, 'World'),
      entity(2, 'mountain', 'world', 0, 'Alpine Mountain'),
      entity(3, 'camera', '', 1, 'Camera'),
    ]);
    const filtered = filterSceneTree(tree, 'alpine');
    expect(filtered).toHaveLength(1);
    expect(filtered[0].name).toBe('World');
    expect(filtered[0].children?.[0].name).toBe('Alpine Mountain');
  });

  it('ignores repeated selection and routes only relevant host refreshes', () => {
    const selected = '7:1';
    expect(classifyHostEventRefresh({ type: 'entity.selected', entity: { index: 7, generation: 1 } }, selected)).toBe('none');
    expect(classifyHostEventRefresh({ type: 'entity.selected', entity: { index: 8, generation: 1 } }, selected)).toBe('selection');
    expect(classifyHostEventRefresh({ type: 'component.changed', entity: { index: 7, generation: 1 } }, selected)).toBe('selected');
    expect(classifyHostEventRefresh({ type: 'component.changed', entity: { index: 3, generation: 1 } }, selected)).toBe('hierarchy');
    expect(classifyHostEventRefresh({ type: 'component.changed', entity: { index: 0xffffffff, generation: 0 } }, selected)).toBe('none');
    expect(classifyHostEventRefresh({ type: 'scene.changed' }, selected)).toBe('all');
    expect(classifyHostEventRefresh({ type: 'viewport.error' }, selected)).toBe('none');
  });
});
