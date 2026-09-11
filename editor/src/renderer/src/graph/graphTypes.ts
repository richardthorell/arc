export type GraphPoint = [number, number];

export type GraphViewport = {
  x: number;
  y: number;
  zoom: number;
};

export type GraphPinDirection = 'input' | 'output';

export type GraphPinDefinition<TPinType = string> = {
  id: string;
  label: string;
  type: TPinType;
};

export type GraphNodeDefinition<
  TNodeType extends string = string,
  TPinType = string,
  TCategory extends string = string,
  TSubcategory extends string = string,
> = {
  type: TNodeType;
  title: string;
  category: TCategory;
  subcategory?: TSubcategory;
  inputs: GraphPinDefinition<TPinType>[];
  outputs: GraphPinDefinition<TPinType>[];
};

export type GraphNodeLike<TNodeType extends string = string> = {
  id: string;
  type: TNodeType;
  position: GraphPoint;
};

export type GraphPinRef = {
  nodeId: string;
  pin: string;
};

export type GraphConnectionLike = {
  id: string;
  from: GraphPinRef;
  to: GraphPinRef;
};

export type GraphConnectionEndpoint<TNode extends GraphNodeLike, TPinType = string> = {
  node: TNode;
  pin: GraphPinDefinition<TPinType>;
  direction: GraphPinDirection;
};

export type GraphConnectionValidation =
  | { allowed: true }
  | {
      allowed: false;
      reason?: string;
    };

export interface GraphDomain<
  TNode extends GraphNodeLike,
  TNodeType extends string = TNode['type'],
  TPinType = string,
  TCategory extends string = string,
  TSubcategory extends string = string,
> {
  getNodeDefinition(node: TNode): GraphNodeDefinition<TNodeType, TPinType, TCategory, TSubcategory>;
  getNodeDefinitions(): readonly GraphNodeDefinition<TNodeType, TPinType, TCategory, TSubcategory>[];
  canConnect(
    from: GraphConnectionEndpoint<TNode, TPinType>,
    to: GraphConnectionEndpoint<TNode, TPinType>,
  ): GraphConnectionValidation;
  canDeleteNode(node: TNode): boolean;
}
