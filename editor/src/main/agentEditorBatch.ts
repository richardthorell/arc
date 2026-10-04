import { z } from 'zod';

const entityGuid = z.string().min(1);
const tempId = z.string().regex(/^[A-Za-z][A-Za-z0-9_-]{0,63}$/u, 'tempId must be a simple local identifier');
const vector3 = z.tuple([z.number(), z.number(), z.number()]);
const quaternion = z.tuple([z.number(), z.number(), z.number(), z.number()]);
const jsonRecord = z.record(z.string(), z.unknown());

export const agentBatchEntityTargetSchema = z.union([
  z.object({ guid: entityGuid }).strict(),
  z.object({ tempId }).strict(),
]);

export type AgentBatchEntityTarget = z.infer<typeof agentBatchEntityTargetSchema>;

const entityCreate = z
  .object({
    type: z.literal('entity.create'),
    tempId: tempId.optional(),
    kind: z.string().min(1).optional(),
    parent: agentBatchEntityTargetSchema.optional(),
  })
  .strict();
const entityRename = z
  .object({ type: z.literal('entity.rename'), target: agentBatchEntityTargetSchema, name: z.string() })
  .strict();
const entitySetActive = z
  .object({ type: z.literal('entity.setActive'), target: agentBatchEntityTargetSchema, active: z.boolean() })
  .strict();
const entitySetTag = z
  .object({ type: z.literal('entity.setTag'), target: agentBatchEntityTargetSchema, tag: z.string() })
  .strict();
const entitySetMobility = z
  .object({ type: z.literal('entity.setMobility'), target: agentBatchEntityTargetSchema, mobility: z.string().min(1) })
  .strict();
const entitySetTransform = z
  .object({
    type: z.literal('entity.setTransform'),
    target: agentBatchEntityTargetSchema,
    transform: z
      .object({ position: vector3, rotation: quaternion, scale: vector3 })
      .strict(),
  })
  .strict();
const entitySetRenderLayer = z
  .object({
    type: z.literal('entity.setRenderLayer'),
    target: agentBatchEntityTargetSchema,
    renderLayerMask: z.number().int().nonnegative(),
  })
  .strict();
const entitySetMaterial = z
  .object({ type: z.literal('entity.setMaterial'), target: agentBatchEntityTargetSchema, path: z.string().min(1) })
  .strict();
const entitySetFlow = z
  .object({
    type: z.literal('entity.setFlow'),
    target: agentBatchEntityTargetSchema,
    path: z.string().optional(),
    assetGuid: z.string().optional(),
    enabled: z.boolean().optional(),
  })
  .strict();
const entityOnly = <T extends string>(type: T) =>
  z.object({ type: z.literal(type), target: agentBatchEntityTargetSchema }).strict();
const entityReparent = z
  .object({
    type: z.literal('entity.reparent'),
    target: agentBatchEntityTargetSchema,
    parent: agentBatchEntityTargetSchema.optional(),
    preserveWorld: z.boolean().optional(),
  })
  .strict();
const entityPatchComponent = z
  .object({
    type: z.literal('entity.patchComponent'),
    target: agentBatchEntityTargetSchema,
    component: z.string().min(1),
    fields: jsonRecord,
  })
  .strict();

export const agentEditorBatchOperationSchema = z.discriminatedUnion('type', [
  entityCreate,
  entityRename,
  entitySetActive,
  entitySetTag,
  entitySetMobility,
  entitySetTransform,
  entitySetRenderLayer,
  entitySetMaterial,
  entitySetFlow,
  entityOnly('entity.snapToFloor'),
  entityOnly('entity.delete'),
  entityOnly('entity.duplicate'),
  entityReparent,
  entityPatchComponent,
]);

export type AgentEditorBatchOperation = z.infer<typeof agentEditorBatchOperationSchema>;

const referencedTempIds = (operation: AgentEditorBatchOperation): string[] => {
  const references: string[] = [];
  const add = (target: AgentBatchEntityTarget | undefined) => {
    if (target && 'tempId' in target) references.push(target.tempId);
  };
  if (operation.type === 'entity.create') add(operation.parent);
  else {
    add(operation.target);
    if (operation.type === 'entity.reparent') add(operation.parent);
  }
  return references;
};

export const agentEditorBatchRequestSchema = z
  .object({
    editSessionId: z.string().min(1),
    expectedSceneRevision: z.number().int().positive(),
    operations: z.array(agentEditorBatchOperationSchema).min(1).max(64),
  })
  .strict()
  .superRefine((request, context) => {
    const available = new Set<string>();
    for (const [index, operation] of request.operations.entries()) {
      for (const reference of referencedTempIds(operation)) {
        if (!available.has(reference)) {
          context.addIssue({
            code: 'custom',
            path: ['operations', index],
            message: `tempId '${reference}' must reference an entity created earlier in this batch`,
          });
        }
      }
      if (operation.type === 'entity.create' && operation.tempId) {
        if (available.has(operation.tempId)) {
          context.addIssue({
            code: 'custom',
            path: ['operations', index, 'tempId'],
            message: `tempId '${operation.tempId}' is already defined`,
          });
        } else {
          available.add(operation.tempId);
        }
      }
    }
  });

export type AgentEditorBatchRequest = z.infer<typeof agentEditorBatchRequestSchema>;

export const parseAgentEditorBatchRequest = (value: unknown): AgentEditorBatchRequest =>
  agentEditorBatchRequestSchema.parse(value);
