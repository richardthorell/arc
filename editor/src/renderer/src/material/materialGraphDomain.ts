import type { GraphConnectionEndpoint, GraphDomain } from '../graph';
import {
  materialNodeDefinition,
  materialNodeDefinitions,
  type MaterialGraphNode,
  type MaterialGraphNodeType,
  type MaterialGraphPinType,
  type MaterialNodeCategory,
  type MaterialNodeSubcategory,
} from './materialGraphTypes';

const materialDefinitions = () =>
  Object.values(materialNodeDefinitions) as Array<(typeof materialNodeDefinitions)[MaterialGraphNodeType]>;

const numericPinTypes = new Set<MaterialGraphPinType>(['float', 'vec2', 'vec3', 'vec4', 'numeric']);

export const materialGraphPinTypesCompatible = (from: MaterialGraphPinType, to: MaterialGraphPinType) => {
  if (from === 'texture2d' || to === 'texture2d') return from === to;
  if (from === 'numeric' || to === 'numeric') return numericPinTypes.has(from) && numericPinTypes.has(to);
  return from === to;
};

export const materialGraphDomain: GraphDomain<
  MaterialGraphNode,
  MaterialGraphNodeType,
  MaterialGraphPinType,
  MaterialNodeCategory,
  MaterialNodeSubcategory
> = {
  getNodeDefinition: materialNodeDefinition,
  getNodeDefinitions: materialDefinitions,
  canConnect: (
    from: GraphConnectionEndpoint<MaterialGraphNode, MaterialGraphPinType>,
    to: GraphConnectionEndpoint<MaterialGraphNode, MaterialGraphPinType>,
  ) => {
    if (from.direction !== 'output' || to.direction !== 'input')
      return { allowed: false, reason: 'Connections must run from an output pin to an input pin.' };
    if (from.node.id === to.node.id) return { allowed: false, reason: 'Material nodes cannot connect to themselves.' };
    if (!materialGraphPinTypesCompatible(from.pin.type, to.pin.type))
      return { allowed: false, reason: `Cannot connect ${from.pin.type} to ${to.pin.type}.` };
    return { allowed: true };
  },
  canDeleteNode: (node) => node.type !== 'output',
};
