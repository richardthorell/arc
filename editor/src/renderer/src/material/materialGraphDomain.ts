import type { GraphConnectionEndpoint, GraphDomain } from '../graph';
import {
  materialNodeDefinitions,
  type MaterialGraphNode,
  type MaterialGraphNodeType,
  type MaterialGraphPinType,
  type MaterialNodeCategory,
  type MaterialNodeSubcategory,
} from './materialGraphTypes';

const materialDefinitions = () =>
  Object.values(materialNodeDefinitions) as Array<(typeof materialNodeDefinitions)[MaterialGraphNodeType]>;

export const materialGraphDomain: GraphDomain<
  MaterialGraphNode,
  MaterialGraphNodeType,
  MaterialGraphPinType,
  MaterialNodeCategory,
  MaterialNodeSubcategory
> = {
  getNodeDefinition: (node) => materialNodeDefinitions[node.type],
  getNodeDefinitions: materialDefinitions,
  canConnect: (
    from: GraphConnectionEndpoint<MaterialGraphNode, MaterialGraphPinType>,
    to: GraphConnectionEndpoint<MaterialGraphNode, MaterialGraphPinType>,
  ) => {
    if (from.direction !== 'output' || to.direction !== 'input')
      return { allowed: false, reason: 'Connections must run from an output pin to an input pin.' };
    if (from.node.id === to.node.id) return { allowed: false, reason: 'Material nodes cannot connect to themselves.' };
    return { allowed: true };
  },
  canDeleteNode: (node) => node.type !== 'output',
};
