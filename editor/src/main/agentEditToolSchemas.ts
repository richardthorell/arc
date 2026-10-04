import { z } from 'zod';

import { agentEditActions, type AgentEditAction } from './agentHarnessContract';

const entityGuid = z.string().min(1);
const vector3 = z.tuple([z.number(), z.number(), z.number()]);
const quaternion = z.tuple([z.number(), z.number(), z.number(), z.number()]);
const jsonRecord = z.record(z.string(), z.unknown());

const transform = z
  .object({
    position: vector3,
    rotation: quaternion,
    scale: vector3,
  })
  .strict();

const entityOnly = z.object({ guid: entityGuid }).strict();

export const agentEditValueSchemas = {
  create: z
    .object({
      kind: z.string().min(1).optional(),
      parentGuid: entityGuid.optional(),
    })
    .strict()
    .describe('create: optional entity kind and parentGuid; no guid is required for the new entity'),
  rename: z
    .object({ guid: entityGuid, name: z.string() })
    .strict()
    .describe('rename: target entity guid and replacement name'),
  setActive: z
    .object({ guid: entityGuid, active: z.boolean() })
    .strict()
    .describe('setActive: target entity guid and active state'),
  setTag: z.object({ guid: entityGuid, tag: z.string() }).strict().describe('setTag: target entity guid and tag'),
  setMobility: z
    .object({ guid: entityGuid, mobility: z.string().min(1) })
    .strict()
    .describe('setMobility: target entity guid and mobility value'),
  setTransform: z
    .object({ guid: entityGuid, transform })
    .strict()
    .describe('setTransform: target entity guid plus nested transform { position, rotation, scale }'),
  setRenderLayer: z
    .object({ guid: entityGuid, renderLayerMask: z.number().int().nonnegative() })
    .strict()
    .describe('setRenderLayer: target entity guid and renderLayerMask'),
  setMaterial: z
    .object({ guid: entityGuid, path: z.string().min(1) })
    .strict()
    .describe('setMaterial: target entity guid and project-relative material path'),
  setFlow: z
    .object({
      guid: entityGuid,
      path: z.string().optional(),
      assetGuid: z.string().optional(),
      enabled: z.boolean().optional(),
    })
    .strict()
    .describe('setFlow: target entity guid with optional flow path, assetGuid, and enabled state'),
  snapToFloor: entityOnly.describe('snapToFloor: target entity guid'),
  delete: entityOnly.describe('delete: target entity guid'),
  duplicate: entityOnly.describe('duplicate: target entity guid'),
  reparent: z
    .object({
      guid: entityGuid,
      parentGuid: entityGuid.optional(),
      preserveWorld: z.boolean().optional(),
    })
    .strict()
    .describe('reparent: target entity guid, optional parentGuid, and optional preserveWorld flag'),
  patchComponent: z
    .object({
      guid: entityGuid,
      component: z.string().min(1),
      fields: jsonRecord,
    })
    .strict()
    .describe('patchComponent: target entity guid, component name, and component fields to merge'),
  createAsset: z
    .object({
      kind: z.string().min(1),
      path: z.string().min(1),
      definition: jsonRecord.optional(),
      source: z.string().optional(),
    })
    .strict()
    .describe('createAsset: asset kind and project-relative path, with definition or source as appropriate'),
  createPrefab: z
    .object({ rootGuid: entityGuid, path: z.string().min(1) })
    .strict()
    .describe('createPrefab: rootGuid and project-relative .arcprefab path'),
  instantiatePrefab: z
    .object({ path: z.string().min(1), parentGuid: entityGuid.optional() })
    .strict()
    .describe('instantiatePrefab: project-relative .arcprefab path and optional parentGuid'),
} as const satisfies Record<AgentEditAction, z.ZodType<unknown>>;

const valueVariants = agentEditActions.map((action) => agentEditValueSchemas[action]);
type AgentEditValueVariants = [
  z.ZodType<unknown>,
  z.ZodType<unknown>,
  ...z.ZodType<unknown>[],
];

export const agentEditValueSchema = z
  .union(valueVariants as unknown as AgentEditValueVariants)
  .describe('Value shape depends on action; use the matching action-specific object shape.');

export const validateAgentEditValue = (action: AgentEditAction, value: unknown): unknown =>
  agentEditValueSchemas[action].parse(value);
