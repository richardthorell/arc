import type { IconType } from 'react-icons';
import { FaAndroid, FaApple, FaLinux, FaPlaystation, FaWindows, FaXbox } from 'react-icons/fa';
import { SiNintendoswitch } from 'react-icons/si';

import type { EditorTargetPlatform } from './MainToolbar';

const platformIcons: Record<EditorTargetPlatform, IconType> = {
  windows: FaWindows,
  linux: FaLinux,
  macos: FaApple,
  ios: FaApple,
  android: FaAndroid,
  xbox: FaXbox,
  playstation: FaPlaystation,
  switch: SiNintendoswitch,
};

export function PlatformBrandIcon({ platform }: { platform: EditorTargetPlatform }) {
  const Icon = platformIcons[platform];
  return (
    <span className="platform-brand-icon" data-platform-icon={platform} aria-hidden="true">
      <Icon />
    </span>
  );
}
