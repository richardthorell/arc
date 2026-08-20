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

const defaultMaterialAsset = (name: string): MaterialAssetJson => {
  const surface = {
    baseColor: { r: 0.78, g: 0.8, b: 0.84, a: 1 },
    metallic: 0,
    roughness: 0.62,
    normalScale: 1,
    aoStrength: 1,
    emissive: { r: 0, g: 0, b: 0 },
    emissiveStrength: 0,
    alphaCutoff: 0.5,
  };
  return {
    version: 3,
    name,
    shader: 'arc/default_phong',
    domain: 'surface',
    blendMode: 'opaque',
    shadingModel: 'standard',
    doubleSided: false,
    surface,
    textures: {
      baseColor: '',
      metallicRoughness: '',
      normal: '',
      ao: '',
      emissive: '',
      height: '',
    },
    advanced: {
      clearCoat: 0,
      sheen: 0,
      transmission: 0,
      subsurface: 0,
      anisotropy: 0,
      parallaxHeightScale: 0,
    },
    graph: createDefaultMaterialGraph({ surface }),
  };
};

const shaderTemplateSource = (template: ShaderAssetTemplate) => {
  if (template === 'compute') {
    return `#version 450\n\nlayout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;\n\nvoid main()\n{\n    // Compute shader entry point.\n}\n`;
  }
  if (template === 'unlit') {
    return `#version 450\n\nlayout(location = 0) in vec3 in_normal;\nlayout(location = 1) in vec3 in_world_position;\nlayout(location = 2) in vec4 in_color;\nlayout(location = 3) in vec2 in_texcoord;\n\nlayout(location = 0) out vec4 out_color;\n\nvoid main()\n{\n    out_color = in_color;\n}\n`;
  }
  if (template === 'post-process') {
    return `#version 450\n\nlayout(set = 0, binding = 0) uniform sampler2D source_color;\nlayout(location = 0) in vec2 in_texcoord;\nlayout(location = 0) out vec4 out_color;\n\nvoid main()\n{\n    out_color = texture(source_color, in_texcoord);\n}\n`;
  }
  if (template === 'empty') {
    return `#version 450\n\nlayout(location = 0) out vec4 out_color;\n\nvoid main()\n{\n    out_color = vec4(1.0, 0.0, 1.0, 1.0);\n}\n`;
  }
  return `#version 450\n\nlayout(location = 0) in vec3 in_normal;\nlayout(location = 1) in vec3 in_world_position;\nlayout(location = 2) in vec4 in_color;\nlayout(location = 3) in vec2 in_texcoord;\n\nlayout(location = 0) out vec4 out_color;\n\nvoid main()\n{\n    vec3 normal = normalize(in_normal);\n    vec3 light_direction = normalize(vec3(0.4, 0.8, 0.3));\n    float diffuse = max(dot(normal, light_direction), 0.0);\n    vec3 lighting = vec3(0.12) + vec3(0.88) * diffuse;\n    out_color = vec4(in_color.rgb * lighting, in_color.a);\n}\n`;
};

export const buildAssetCreation = (
  project: Pick<ProjectSnapshot, 'root' | 'assetRoot'>,
  request: AssetCreationRequest,
): AssetCreationDefinition => {
  const name = cleanAssetName(request.name);
  const folder = request.folder || projectAssetRootPath(project);
  const extension = request.kind === 'material' ? 'arcmat' : request.template === 'compute' ? 'comp' : 'frag';
  const path = joinPath(folder, `${name}.${extension}`);
  const contents =
    request.kind === 'material'
      ? `${JSON.stringify(defaultMaterialAsset(name), null, 2)}\n`
      : shaderTemplateSource(request.template);
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
