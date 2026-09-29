import { flowNodeDefinitions, type FlowGraphConnection, type FlowGraphNode, type FlowNodeType } from './flowGraphTypes';

export type FlowExecutionStep = {
  connectionId: string;
  outputPinId: string;
  targetNodeId: string;
  targetPinId: string;
  ordinal: number;
};

/**
 * Returns execution output pins in the order authored by the Flow language definition.
 *
 * Keeping this policy separate from graph layout makes fan-out semantics explicit and
 * deterministic for Sequence, Switch, Timer, and future multi-output control nodes.
 */
export const getFlowExecutionOutputOrder = (nodeType: FlowNodeType): readonly string[] =>
  flowNodeDefinitions[nodeType].outputs
    .filter((pin) => pin.type.kind === 'execution')
    .map((pin) => pin.id);

/**
 * Resolves outgoing execution edges in language order rather than canvas or insertion order.
 * Multiple edges on the same output are ordered by stable connection id as a deterministic
 * fallback until the compiler rejects or assigns domain-specific semantics to that shape.
 */
export const resolveFlowExecutionOrder = (
  node: Pick<FlowGraphNode, 'id' | 'type'>,
  connections: readonly FlowGraphConnection[],
): FlowExecutionStep[] => {
  const pinOrder = new Map(getFlowExecutionOutputOrder(node.type).map((pinId, ordinal) => [pinId, ordinal]));

  return connections
    .filter((connection) => connection.kind === 'execution' && connection.from.nodeId === node.id)
    .flatMap((connection) => {
      const ordinal = pinOrder.get(connection.from.pin);
      if (ordinal === undefined) return [];
      return [{
        connectionId: connection.id,
        outputPinId: connection.from.pin,
        targetNodeId: connection.to.nodeId,
        targetPinId: connection.to.pin,
        ordinal,
      }];
    })
    .sort((left, right) => left.ordinal - right.ordinal || left.connectionId.localeCompare(right.connectionId));
};
