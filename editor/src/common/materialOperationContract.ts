import { z } from 'zod';

const guid = z.string().regex(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/iu);
const unsignedId = z.string().regex(/^(0|[1-9][0-9]*)$/u);
const parameterType = z.enum(['float', 'vec2', 'vec3', 'vec4', 'texture2d', 'unsupported']);
const assetReference = z.object({ guid, pathHint: z.string() }).strict();
const parameterValue = z.union([z.number(), z.array(z.number()).max(4), z.string(), assetReference, z.null()]);
const pin = z
  .object({
    id: z.string(),
    name: z.string(),
    type: parameterType,
    hasDefault: z.boolean(),
    defaultValue: z.array(z.number()).max(4).nullable(),
  })
  .strict();

export const assetInspectInput = z.object({ guid }).strict();
export const assetInspectOutput = z
  .object({
    guid,
    typeId: z.string(),
    title: z.string(),
    sourcePath: z.string(),
    scope: z.enum(['project', 'builtin']),
    readOnly: z.boolean(),
    generation: unsignedId,
    revision: unsignedId,
    catalogRevision: unsignedId,
    activeProvider: z.string(),
    providers: z.array(
      z
        .object({
          id: z.string(),
          kind: z.enum(['project-source', 'builtin-source', 'virtual-builtin', 'cooked-package', 'runtime-overlay']),
          active: z.boolean(),
          readOnly: z.boolean(),
          generation: unsignedId,
        })
        .strict(),
    ),
    sourceMissing: z.boolean(),
    hasLastGood: z.boolean(),
    dependencies: z.array(guid),
    diagnostics: z.array(z.string()),
  })
  .strict();

export const materialDescribeInput = assetInspectInput;
export const materialEntityInspectInput = z.object({ entityGuid: guid }).strict();
export const materialDescribeOutput = z
  .object({
    guid,
    kind: z.enum(['material', 'materialInstance', 'materialFunction']),
    provenance: z.literal('savedSource'),
    definitionRevision: z.string().regex(/^[0-9a-f]{64}$/u),
    parentGuid: guid.nullable(),
    dependencies: z.array(assetReference),
    parameters: z.array(
      z
        .object({
          id: unsignedId,
          name: z.string(),
          type: parameterType,
          defaultValue: parameterValue,
          inheritedValue: parameterValue,
          overridden: z.boolean(),
          slotId: z.string(),
          semantic: z.enum(['none', 'baseColor']),
          resolvedValue: parameterValue,
          valueStatus: z.enum(['resolved', 'unavailable']),
          range: z.object({ min: z.number(), max: z.number() }).strict().nullable(),
        })
        .strict(),
    ),
    functionSlots: z.array(
      z
        .object({
          id: z.string(),
          name: z.string(),
          defaultFunctionGuid: guid,
          selectedFunctionGuid: guid,
          inputs: z.array(pin),
          outputs: z.array(pin),
          alternatives: z.array(assetReference).max(16),
          alternativesTruncated: z.boolean(),
        })
        .strict(),
    ),
    functionInterface: z
      .object({ inputs: z.array(pin), outputs: z.array(pin) })
      .strict()
      .nullable(),
    diagnostics: z.array(z.object({ code: z.string(), message: z.string() }).strict()),
    definitionReady: z.boolean(),
    runtimeStatus: z.literal('notInspected'),
    entity: z
      .object({
        guid,
        materialGuid: guid,
        runtimeHandle: z
          .object({ index: z.number().int().nonnegative(), generation: z.number().int().nonnegative() })
          .strict(),
        resourceAlive: z.boolean(),
      })
      .strict()
      .nullable(),
  })
  .strict();

export type MaterialDescription = z.infer<typeof materialDescribeOutput>;
export type AssetInspection = z.infer<typeof assetInspectOutput>;

const materialEditAuthority = z
  .object({
    editSessionId: z.string().min(1),
    entityGuid: guid,
    expectedSceneRevision: z.number().int().positive(),
    worldEpoch: z.number().int().nonnegative(),
    definitionRevision: z.string().regex(/^[0-9a-f]{64}$/u),
  })
  .strict();
export const materialEditInputs = {
  assign: materialEditAuthority.extend({ assetGuid: guid }),
  setOverride: materialEditAuthority.extend({
    parameterId: unsignedId,
    value: z.union([z.number(), z.array(z.number()).min(1).max(4), assetReference]),
  }),
  resetOverride: materialEditAuthority.extend({ parameterId: unsignedId }),
  selectFunction: materialEditAuthority.extend({ slotId: z.string().min(1).max(256), assetGuid: guid }),
};
export const materialEditOutput = z
  .object({
    entityGuid: guid,
    applied: z.literal(true),
    visualVerified: z.literal(false),
    sceneRevision: z.number().int().nonnegative(),
    worldEpoch: z.number().int().nonnegative(),
    frameRevision: z.number().int().nonnegative(),
  })
  .strict();
