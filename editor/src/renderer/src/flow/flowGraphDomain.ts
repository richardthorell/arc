import type { GraphConnectionEndpoint, GraphDomain } from '../graph';
import {
  flowNodeDefinitions,
  type FlowGraphNode,
  type FlowNodeCategory,
  type FlowNodeSubcategory,
  type FlowNodeType,
  type FlowPinType,
} from './flowGraphTypes';

const flowDefinitions = () => Object.values(flowNodeDefinitions);

export const flowPinTypesCompatible = (from: FlowPinType, to: FlowPinType) => {
  if (from.kind !== to.kind) return false;
  if (from.kind === 'execution' && to.kind === 'execution') return true;
  if (from.kind !== 'value' || to.kind !== 'value') return false;
  return from.valueType === 'any' || to.valueType === 'any' || from.valueType === to.valueType;
};

export const flowGraphDomain: GraphDomain<
  FlowGraphNode,
  FlowNodeType,
  FlowPinType,
  FlowNodeCategory,
  FlowNodeSubcategory
> = {
  getNodeDefinition: (node) => flowNodeDefinitions[node.type],
  getNodeDefinitions: flowDefinitions,
  canConnect: (
    from: GraphConnectionEndpoint<FlowGraphNode, FlowPinType>,
    to: GraphConnectionEndpoint<FlowGraphNode, FlowPinType>,
  ) => {
    if (from.direction !== 'output' || to.direction !== 'input')
      return { allowed: false, reason: 'Flow connections must run from an output pin to an input pin.' };
    if (from.node.id === to.node.id) return { allowed: false, reason: 'A Flow node cannot connect to itself.' };
    if (!flowPinTypesCompatible(from.pin.type, to.pin.type))
      return { allowed: false, reason: 'The Flow pin types are not compatible.' };
    return { allowed: true };
  },
  canDeleteNode: () => true,
};
