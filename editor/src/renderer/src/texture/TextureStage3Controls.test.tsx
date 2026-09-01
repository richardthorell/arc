// @vitest-environment jsdom

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { TextureStage3Controls } from './TextureStage3Controls';

const settings = {
  settingsVersion: 5,
  preset: 'color' as const,
  semantic: 'base_color' as const,
  colorSpace: 'linear' as const,
  streamingMode: 'streamed_mips' as const,
  compression: 'color' as const,
  powerOfTwo: 'preserve' as const,
  minFilter: 'linear' as const,
  magFilter: 'linear' as const,
  mipFilter: 'linear' as const,
  wrapU: 'repeat' as const,
  wrapV: 'repeat' as const,
  mipGenerationFilter: 'box' as const,
  maxSize: 8192,
  anisotropy: 8,
  lodBias: 0,
  minimumLod: 0,
  maximumLod: 1000,
  alphaCoverageThreshold: 0.5,
  brightness: 0.2,
  gamma: 1.55,
  contrast: 1,
  saturation: 1,
  vibrance: 0,
  tintR: 1,
  tintG: 1,
  tintB: 1,
  inputBlack: 0,
  inputWhite: 1,
  outputBlack: 0,
  outputWhite: 1,
  channelR: 'red' as const,
  channelG: 'green' as const,
  channelB: 'blue' as const,
  channelA: 'alpha' as const,
  invertR: false,
  invertG: false,
  invertB: false,
  invertA: false,
  generateMips: true,
  preserveAlphaCoverage: false,
};

describe('TextureStage3Controls', () => {
  afterEach(() => vi.restoreAllMocks());

  it('previews slider changes immediately and resets controls to their neutral default', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true });
    const query = vi.fn().mockResolvedValue({ succeeded: true, payload: settings });
    Object.defineProperty(window, 'arc', { configurable: true, value: { host: { command, query } } });
    const preview = vi.fn();
    window.addEventListener('arc:texture-settings-preview', preview);

    render(
      <TextureStage3Controls
        asset={{ id: 'texture', name: 'T', path: 'T.png', kind: 'texture', status: 'ready', guid: 'guid' }}
      />,
    );
    const slider = await screen.findByLabelText('Brightness slider');
    fireEvent.change(slider, { target: { value: '0.6' } });
    expect(preview).toHaveBeenCalled();
    expect((screen.getByRole('spinbutton', { name: 'Brightness' }) as HTMLInputElement).value).toBe('0.6');

    fireEvent.click(screen.getByLabelText('Reset Brightness'));
    expect((screen.getByRole('spinbutton', { name: 'Brightness' }) as HTMLInputElement).value).toBe('0');
    await waitFor(() => expect(command).toHaveBeenCalled());
    window.removeEventListener('arc:texture-settings-preview', preview);
  });
});
