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
    sourceMissing: z.boolean(),
    hasLastGood: z.boolean(),
    dependencies: z.array(guid),
    diagnostics: z.array(z.string()),
  })
  .strict();

export const materialDescribeInput = assetInspectInput;
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
  })
  .strict();

export type MaterialDescription = z.infer<typeof materialDescribeOutput>;
export type AssetInspection = z.infer<typeof assetInspectOutput>;
