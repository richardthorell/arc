import { createDefaultMaterialGraph, type MaterialAssetJson } from '../material/materialGraphTypes';
import type { AssetItem, ProjectSnapshot } from '../services/editorHostTypes';

export type ShaderAssetTemplate = 'surface' | 'unlit' | 'compute' | 'post-process' | 'empty';

export type AssetCreationRequest =
  | { kind: 'material'; name: string; folder: string }
  | { kind: 'shader'; name: string; folder: string; template: ShaderAssetTemplate };

export type AssetCreationDefinition = {
  asset: AssetItem;
  contents: string;
};

const normalizePath = (value: string) => value.replaceAll('\\', '/').replace(/\/+$/, '');

export const projectAssetRootPath = (project: Pick<ProjectSnapshot, 'root' | 'assetRoot'>) => {
  const projectRoot = normalizePath(project.root);
  const assetRoot = normalizePath(project.assetRoot);
  if (!assetRoot) return 'Content';
  if (projectRoot && assetRoot.toLocaleLowerCase().startsWith(`${projectRoot.toLocaleLowerCase()}/`)) {
    return assetRoot.slice(projectRoot.length + 1);
  }
  if (!/^[a-z]:\//i.test(assetRoot) && !assetRoot.startsWith('/')) return assetRoot;
  return assetRoot.split('/').at(-1) || 'Content';
};

const cleanAssetName = (name: string) => {
  const value = name.trim().replace(/\.(arcmat|frag|vert|comp)$/i, '');
  if (!value) throw new Error('Enter a name for the asset');
  if (value === '.' || value === '..' || /[<>:"/\\|?*]/.test(value)) {
    throw new Error('Asset names cannot contain path or reserved file-system characters');
  }
  return value;
};

const joinPath = (folder: string, fileName: string) => `${normalizePath(folder)}/${fileName}`.replace(/^\//, '');

const defaultMaterialAsset = (name: string): MaterialAssetJson => ({
  version: 4,
  name,
  domain: 'surface',
  blendMode: 'opaque',
  shadingModel: 'standard',
  doubleSided: false,
  graph: createDefaultMaterialGraph(),
});

const shaderTemplateSource = (template: ShaderAssetTemplate) => {
  if (template === 'compute') {
    return `[shader("compute")]\n[numthreads(8, 8, 1)]\nvoid main(uint3 dispatchThreadId : SV_DispatchThreadID)\n{\n    // Add backend-neutral compute work here.\n}\n`;
  }
  if (template === 'unlit') {
    return `struct SurfaceInput { float4 color : COLOR0; };\n\n[shader("fragment")]\nfloat4 main(SurfaceInput input) : SV_Target\n{\n    return input.color;\n}\n`;
  }
  if (template === 'post-process') {
    return `Texture2D<float4> sourceColor;\nSamplerState sourceSampler;\n\nstruct PostInput { float2 uv : TEXCOORD0; };\n\n[shader("fragment")]\nfloat4 main(PostInput input) : SV_Target\n{\n    return sourceColor.Sample(sourceSampler, input.uv);\n}\n`;
  }
  if (template === 'empty') {
    return `[shader("fragment")]\nfloat4 main() : SV_Target\n{\n    return float4(1.0, 0.0, 1.0, 1.0);\n}\n`;
  }
  return `struct SurfaceInput\n{\n    float3 normalWS : TEXCOORD0;\n    float4 color : COLOR0;\n};\n\n[shader("fragment")]\nfloat4 main(SurfaceInput input) : SV_Target\n{\n    float3 normal = normalize(input.normalWS);\n    float3 lightDirection = normalize(float3(0.4, 0.8, 0.3));\n    float diffuse = max(dot(normal, lightDirection), 0.0);\n    float3 lighting = float3(0.12) + float3(0.88) * diffuse;\n    return float4(input.color.rgb * lighting, input.color.a);\n}\n`;
};

export const buildAssetCreation = (
  project: Pick<ProjectSnapshot, 'root' | 'assetRoot'>,
  request: AssetCreationRequest,
): AssetCreationDefinition => {
  const name = cleanAssetName(request.name);
  const folder = request.folder || projectAssetRootPath(project);
  const extension = request.kind === 'material' ? 'arcmat' : 'slang';
  const path = joinPath(folder, `${name}.${extension}`);
  const contents =
    request.kind === 'material'
      ? `${JSON.stringify(defaultMaterialAsset(name), null, 2)}\n`
      : shaderTemplateSource(request.template);

  if (request.kind === 'material' || request.kind === 'shader') {
    console.info('[material-flow] asset creation path', {
      kind: request.kind,
      projectRoot: project.root,
      assetRoot: project.assetRoot,
      requestedFolder: request.folder,
      resolvedFolder: folder,
      path,
    });
  }

  return {
    asset: {
      id: path,
      name: `${name}.${extension}`,
      path,
      scope: 'project',
      readOnly: false,
      kind: request.kind,
      status: 'ready',
    },
    contents,
  };
};
