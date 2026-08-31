import type { CSSProperties } from 'react';

import documentTypeIconAtlasUrl from './document-type-icons-atlas.png';

export type DocumentTypeIconKind =
  | 'world'
  | 'level'
  | 'scene'
  | 'material'
  | 'shader'
  | 'image'
  | 'texture'
  | 'mesh'
  | 'prefab'
  | 'animation'
  | 'audio'
  | 'script'
  | 'font'
  | 'folder'
  | 'settings';

const atlasColumns = 4;
const atlasRows = 3;

const iconCells: Record<DocumentTypeIconKind, readonly [number, number]> = {
  world: [0, 0],
  level: [0, 0],
  scene: [0, 0],
  material: [1, 0],
  shader: [2, 0],
  image: [3, 0],
  texture: [3, 0],
  mesh: [0, 1],
  prefab: [1, 1],
  animation: [2, 1],
  audio: [3, 1],
  script: [0, 2],
  font: [1, 2],
  folder: [2, 2],
  settings: [3, 2],
};

export const documentTypeIconCell = (kind: DocumentTypeIconKind) => iconCells[kind];

export function DocumentTypeIcon({
  kind,
  size = 16,
  className,
  title,
  style,
}: {
  kind: DocumentTypeIconKind;
  size?: number;
  className?: string;
  title?: string;
  style?: CSSProperties;
}) {
  const [column, row] = documentTypeIconCell(kind);

  return (
    <span
      aria-hidden={title ? undefined : true}
      aria-label={title}
      className={className}
      data-document-type-icon={kind}
      role={title ? 'img' : undefined}
      style={{
        display: 'inline-block',
        width: size,
        height: size,
        flex: `0 0 ${size}px`,
        overflow: 'hidden',
        position: 'relative',
        ...style,
      }}
    >
      <img
        alt=""
        aria-hidden="true"
        data-document-type-icon-image=""
        draggable={false}
        src={documentTypeIconAtlasUrl}
        style={{
          position: 'absolute',
          width: size * atlasColumns,
          height: size * atlasRows,
          maxWidth: 'none',
          left: -column * size,
          top: -row * size,
          pointerEvents: 'none',
          userSelect: 'none',
        }}
      />
    </span>
  );
}
