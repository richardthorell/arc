import { forwardRef, type InputHTMLAttributes, type TextareaHTMLAttributes } from 'react';

type UiTextInputProps = InputHTMLAttributes<HTMLInputElement>;
type UiTextAreaProps = TextareaHTMLAttributes<HTMLTextAreaElement>;

export const UiTextInput = forwardRef<HTMLInputElement, UiTextInputProps>(function UiTextInput(
  { className, ...props },
  ref,
) {
  return <input ref={ref} className={['ui-text-input', className].filter(Boolean).join(' ')} {...props} />;
});

export const UiSearchInput = forwardRef<HTMLInputElement, UiTextInputProps>(function UiSearchInput(
  { className, type = 'search', ...props },
  ref,
) {
  return (
    <UiTextInput
      ref={ref}
      className={['ui-search-input', className].filter(Boolean).join(' ')}
      type={type}
      {...props}
    />
  );
});

export const UiTextArea = forwardRef<HTMLTextAreaElement, UiTextAreaProps>(function UiTextArea(
  { className, ...props },
  ref,
) {
  return <textarea ref={ref} className={['ui-text-input', 'ui-text-area', className].filter(Boolean).join(' ')} {...props} />;
});
