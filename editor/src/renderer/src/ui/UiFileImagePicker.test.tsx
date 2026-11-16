// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { UiFileImagePicker } from './UiFileImagePicker';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it('center-crops disk images to 1280 by 720 WebP without publishing files', async () => {
  vi.stubGlobal(
    'Image',
    class {
      width = 800;
      height = 800;
      src = '';
      decode() {
        return Promise.resolve();
      }
    },
  );
  const drawImage = vi.fn();
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    drawImage,
  } as unknown as CanvasRenderingContext2D);
  const encode = vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/webp;base64,draft');
  const onChange = vi.fn();
  render(
    <UiFileImagePicker
      browse={vi.fn().mockResolvedValue('data:image/png;base64,input')}
      onChange={onChange}
      onError={vi.fn()}
    />,
  );
  fireEvent.click(screen.getByRole('button', { name: 'Browse' }));
  await waitFor(() => expect(onChange).toHaveBeenCalledWith('data:image/webp;base64,draft'));
  expect(drawImage).toHaveBeenCalledWith(expect.anything(), 0, 175, 800, 450, 0, 0, 1280, 720);
  expect(encode).toHaveBeenCalledWith('image/webp', 0.9);
});

it('keeps the current preview when the native dialog is cancelled', async () => {
  const onChange = vi.fn();
  render(
    <UiFileImagePicker
      preview="data:image/webp;base64,old"
      browse={vi.fn().mockResolvedValue(null)}
      onChange={onChange}
      onError={vi.fn()}
    />,
  );
  fireEvent.click(screen.getByRole('button', { name: 'Replace' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Replace' })).toBeEnabled());
  expect(onChange).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Remove' }));
  expect(onChange).toHaveBeenCalledWith(null);
});

it('discards late dialog results after the settings control unmounts', async () => {
  let finish!: (value: string | null) => void;
  const onChange = vi.fn();
  const view = render(
    <UiFileImagePicker
      browse={() =>
        new Promise((resolve) => {
          finish = resolve;
        })
      }
      onChange={onChange}
      onError={vi.fn()}
    />,
  );
  fireEvent.click(screen.getByRole('button', { name: 'Browse' }));
  view.unmount();
  finish('data:image/png;base64,input');
  await Promise.resolve();
  expect(onChange).not.toHaveBeenCalled();
});
