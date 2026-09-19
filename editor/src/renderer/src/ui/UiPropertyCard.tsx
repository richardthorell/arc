import { forwardRef, type ReactNode } from 'react';

import { UiPanelCard } from './UiPanelCard';
import type { UiPanelCardProps } from './UiPanelCard';
import { UiPanelCardRow } from './UiPanelCardRow';
import type { UiPanelCardRowProps } from './UiPanelCardRow';

export type UiPropertyCardField = {
  id: string;
  label: ReactNode;
  description?: ReactNode;
  control?: ReactNode;
  align?: UiPanelCardRowProps['align'];
  controlClassName?: string;
  className?: string;
  fullWidth?: boolean;
  tooltip?: string;
};

export type UiPropertyCardProps = Omit<UiPanelCardProps, 'children'> & {
  fields: ReadonlyArray<UiPropertyCardField>;
  emptyState?: ReactNode;
  rowsClassName?: string;
};

export const UiPropertyCard = forwardRef<HTMLElement, UiPropertyCardProps>(function UiPropertyCard(
  { fields, emptyState, rowsClassName, ...cardProps },
  ref,
) {
  return (
    <UiPanelCard {...cardProps} ref={ref}>
      <div className={['ui-property-card-rows', rowsClassName].filter(Boolean).join(' ')}>
        {fields.map((field) => (
          <UiPanelCardRow
            align={field.align}
            className={field.className}
            controlClassName={field.controlClassName}
            description={field.description}
            fullWidth={field.fullWidth}
            key={field.id}
            label={field.label}
            title={field.tooltip}
          >
            {field.control}
          </UiPanelCardRow>
        ))}
        {!fields.length && emptyState}
      </div>
    </UiPanelCard>
  );
});
