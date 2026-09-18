import { forwardRef } from 'react';

import { UiPanelCard, type UiPanelCardProps } from './UiPanelCard';

export type UiPanelSectionProps = UiPanelCardProps;

export const UiPanelSection = forwardRef<HTMLElement, UiPanelSectionProps>(function UiPanelSection(props, ref) {
  return <UiPanelCard ref={ref} {...props} />;
});
