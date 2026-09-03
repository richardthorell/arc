import { useMemo } from 'react';

import { ContentAssetCard } from '../content/ContentAssetCard';
import '../content/contentBrowser.css';
import type { AssetThumbnailProvider } from '../inspector/AssetPicker';
import type { AssetItem } from '../services/editorHostTypes';

import './uiLabContentCards.css';

const svgDataUri = (svg: string) => `data:image/svg+xml,${encodeURIComponent(svg)}`;

const texturePreview = svgDataUri(`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 200">
    <defs>
      <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#5b8fac"/>
        <stop offset="0.58" stop-color="#d7a878"/>
        <stop offset="1" stop-color="#e0c59a"/>
      </linearGradient>
      <linearGradient id="ridge" x1="0" y1="0" x2="1" y2="1">
        <stop stop-color="#384957"/>
        <stop offset="1" stop-color="#17232d"/>
      </linearGradient>
    </defs>
    <rect width="320" height="200" fill="url(#sky)"/>
    <circle cx="240" cy="62" r="24" fill="#f3d59b" opacity=".85"/>
    <path d="M0 162 72 91l46 43 42-61 82 89 42-38 36 39v37H0Z" fill="url(#ridge)"/>
    <path d="M0 174 84 130l57 30 49-20 130 38v22H0Z" fill="#233640" opacity=".88"/>
  </svg>
`);

const materialPreview = svgDataUri(`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 200">
    <defs>
      <radialGradient id="sphere" cx="34%" cy="26%" r="72%">
        <stop offset="0" stop-color="#f2c18c"/>
        <stop offset=".28" stop-color="#a65f32"/>
        <stop offset=".72" stop-color="#62351f"/>
        <stop offset="1" stop-color="#2d1b15"/>
      </radialGradient>
      <radialGradient id="shine" cx="50%" cy="50%" r="50%">
        <stop stop-color="#fff" stop-opacity=".75"/>
        <stop offset="1" stop-color="#fff" stop-opacity="0"/>
      </radialGradient>
    </defs>
    <rect width="320" height="200" fill="#161e25"/>
    <ellipse cx="160" cy="166" rx="78" ry="14" fill="#000" opacity=".38"/>
    <circle cx="160" cy="96" r="73" fill="url(#sphere)"/>
    <ellipse cx="132" cy="67" rx="35" ry="22" fill="url(#shine)" opacity=".42"/>
  </svg>
`);

const modelPreview = svgDataUri(`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 200">
    <defs>
      <linearGradient id="top" x1="0" y1="0" x2="1" y2="1">
        <stop stop-color="#dfe5e9"/>
        <stop offset="1" stop-color="#a8b5bf"/>
      </linearGradient>
      <linearGradient id="side" x1="0" y1="0" x2="1" y2="1">
        <stop stop-color="#7e909f"/>
        <stop offset="1" stop-color="#4c606f"/>
      </linearGradient>
    </defs>
    <ellipse cx="163" cy="164" rx="83" ry="12" fill="#000" opacity=".18"/>
    <path d="m82 118 86-57 74 38-86 58Z" fill="url(#top)" stroke="#eaf0f4" stroke-width="3"/>
    <path d="m82 118 74 39v24l-74-40Z" fill="#667b8a" stroke="#8fa0ad" stroke-width="3"/>
    <path d="m156 157 86-58v25l-86 57Z" fill="url(#side)" stroke="#788d9c" stroke-width="3"/>
    <path d="m120 97 47-31 39 20-48 31Z" fill="#eef2f4" opacity=".78"/>
  </svg>
`);

const assets: AssetItem[] = [
  {
    id: 'ui-lab-texture-card',
    name: 'T_Mountain_Sunset.png',
    path: 'Content/Textures/T_Mountain_Sunset.png',
    kind: 'texture',
    status: 'ready',
    scope: 'project',
    sourceBytes: 3_481_220,
    width: 2048,
    height: 2048,
    mipLevels: 12,
    importerId: 'texture',
    residency: 'device',
  },
  {
    id: 'ui-lab-material-card',
    name: 'M_Warm_Wood.arcmat',
    path: 'Content/Materials/M_Warm_Wood.arcmat',
    kind: 'material',
    status: 'ready',
    scope: 'project',
    sourceBytes: 18_432,
    importerId: 'material',
    residency: 'device',
  },
  {
    id: 'ui-lab-model-card',
    name: 'SM_Shipping_Crate.glb',
    path: 'Content/Models/SM_Shipping_Crate.glb',
    kind: 'scene',
    status: 'ready',
    scope: 'project',
    sourceBytes: 1_942_336,
    vertexCount: 12_864,
    triangleCount: 7_244,
    importerId: 'gltf',
    residency: 'derived',
  },
];

const thumbnails = new Map<string, string>([
  [assets[0].path, texturePreview],
  [assets[1].path, materialPreview],
  [assets[2].path, modelPreview],
]);

export function UiLabContentCards() {
  const thumbnailProvider = useMemo<AssetThumbnailProvider>(
    () => async (path) => thumbnails.get(path) ?? null,
    [],
  );

  return (
    <div className="ui-lab-content-card-showcase content-browser-v2">
      <div aria-label="Content Browser card examples" aria-multiselectable="true" className="content-assets grid" role="listbox">
        {assets.map((asset, index) => (
          <ContentAssetCard
            asset={asset}
            favorite={index === 1}
            key={asset.id}
            selected={false}
            thumbnailProvider={thumbnailProvider}
            onActivate={() => undefined}
            onFavorite={() => undefined}
            onReimport={() => undefined}
            onSelect={() => undefined}
          />
        ))}
      </div>
      <p>Hover any card to preview the production asset-details tooltip.</p>
    </div>
  );
}
