import { describe, expect, it } from 'vitest';

import {
  createDefaultFlowGraph,
  createFlowAsset,
  createFlowNode,
  flowGraphFromAsset,
  flowNodeDefinitions,
  isFlowAssetJson,
  isFlowGraph,
} from './flowGraphTypes';

describe('Flow graph authoring schema', () => {
  it('creates a v1 Flow asset with a Begin Play entry point', () => {
    const asset = createFlowAsset('PlayerController');

    expect(asset.assetType).toBe('flow');
    expect(asset.version).toBe(1);
    expect(asset.name).toBe('PlayerController');
    expect(asset.graph.version).toBe(1);
    expect(asset.graph.nodes).toHaveLength(1);
    expect(asset.graph.nodes[0].type).toBe('beginPlay');
    expect(isFlowAssetJson(asset)).toBe(true);
  });

  it('round-trips the authored graph without sharing mutable state', () => {
    const asset = createFlowAsset('Gameplay');
    asset.graph.nodes.push(createFlowNode('branch', [360, 140]));

    const graph = flowGraphFromAsset(asset);
    graph.nodes[0].position[0] = 999;

    expect(asset.graph.nodes[0].position[0]).not.toBe(999);
    expect(isFlowGraph(JSON.parse(JSON.stringify(graph)))).toBe(true);
  });

  it('creates typed gameplay literal defaults', () => {
    expect(createFlowNode('boolLiteral', [0, 0]).values.value).toBe(false);
    expect(createFlowNode('stringLiteral', [0, 0]).values.value).toBe('');
    expect(createFlowNode('vector3Literal', [0, 0]).values.value).toEqual([0, 0, 0]);
    expect(createFlowNode('vector4Literal', [0, 0]).values.value).toEqual([0, 0, 0, 1]);
  });

  it('publishes the F5.1 gameplay node categories', () => {
    expect(flowNodeDefinitions.selfEntity.category).toBe('Entity');
    expect(flowNodeDefinitions.setName.category).toBe('Components');
    expect(flowNodeDefinitions.setTransform.inputs.map((pin) => pin.id)).toEqual([
      'exec',
      'entity',
      'position',
      'rotation',
      'scale',
    ]);
  });

  it('creates structural gameplay nodes with stable defaults', () => {
    const create = createFlowNode('createEntity', [0, 0]);
    const hasComponent = createFlowNode('hasCoreComponent', [200, 0]);
    const removeComponent = createFlowNode('removeCoreComponent', [400, 0]);

    expect(create.values).toEqual({});
    expect(hasComponent.values.component).toBe('transform');
    expect(removeComponent.values.component).toBe('transform');
  });

  it('creates and validates F8.1 graph interface and custom-event authoring data', () => {
    const asset = createFlowAsset('ReusableLogic');
    asset.graph.inputs!.push({ id: 'amount', name: 'Amount', type: 'int', defaultValue: 1 });
    asset.graph.outputs!.push({ id: 'result', name: 'Result', type: 'int', defaultValue: 0 });
    asset.graph.events!.push({ id: 'apply', name: 'Apply' });
    asset.graph.nodes.push(createFlowNode('customEvent', [300, 100], { eventId: 'apply' }));
    asset.graph.nodes.push(createFlowNode('graphInput', [300, 240], { interfaceId: 'amount', interfaceType: 'int' }));
    asset.graph.nodes.push(createFlowNode('graphOutput', [560, 100], { interfaceId: 'result', interfaceType: 'int' }));

    expect(isFlowAssetJson(asset)).toBe(true);
    expect(flowNodeDefinitions.customEvent.category).toBe('Events');
    expect(flowNodeDefinitions.graphInput.category).toBe('Interface');
    expect(flowNodeDefinitions.graphOutput.inputs.map((pin) => pin.id)).toEqual(['exec', 'value']);
  });

  it('creates and validates F8.2 local function declarations', () => {
    const asset = createFlowAsset('Functions');
    asset.graph.functions!.push({
      id: 'identity',
      name: 'Identity',
      inputs: [{ id: 'value', name: 'Value', type: 'int', defaultValue: 0 }],
      outputs: [{ id: 'result', name: 'Result', type: 'int', defaultValue: 0 }],
    });
    asset.graph.nodes.push(
      createFlowNode('functionEntry', [200, 200], {
        functionId: 'identity',
        functionName: 'Identity',
        functionInputs: asset.graph.functions![0].inputs,
        functionOutputs: asset.graph.functions![0].outputs,
      }),
    );

    expect(isFlowAssetJson(asset)).toBe(true);
    expect(flowNodeDefinitions.callFunction.category).toBe('Functions');
    expect(flowNodeDefinitions.functionReturn.subcategory).toBe('Local');
  });

  it('normalizes legacy v1 graphs without F8.1 declaration arrays', () => {
    const asset = createFlowAsset('Legacy');
    delete asset.graph.inputs;
    delete asset.graph.outputs;
    delete asset.graph.events;
    delete asset.graph.functions;

    expect(isFlowAssetJson(asset)).toBe(true);
    const graph = flowGraphFromAsset(asset);
    expect(graph.inputs).toEqual([]);
    expect(graph.outputs).toEqual([]);
    expect(graph.events).toEqual([]);
    expect(graph.functions).toEqual([]);
  });

  it('rejects connections that reference missing nodes', () => {
    const graph = createDefaultFlowGraph();
    graph.connections.push({
      id: 'broken',
      kind: 'execution',
      from: { nodeId: graph.nodes[0].id, pin: 'exec' },
      to: { nodeId: 'missing', pin: 'exec' },
    });

    expect(isFlowGraph(graph)).toBe(false);
  });
});
