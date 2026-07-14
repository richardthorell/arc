import type { InputHTMLAttributes } from 'react';

type UiTextInputProps = InputHTMLAttributes<HTMLInputElement>;

export function UiTextInput({ className, ...props }: UiTextInputProps) {
  return <input className={['ui-text-input', className].filter(Boolean).join(' ')} {...props} />;
}

export function UiSearchInput({ className, type = 'search', ...props }: UiTextInputProps) {
  return <UiTextInput className={['ui-search-input', className].filter(Boolean).join(' ')} type={type} {...props} />;
}
