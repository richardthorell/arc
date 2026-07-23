// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { WorldEnvironmentInspector } from './WorldEnvironmentInspector';
import type { HostCloudLayer, HostWorldEnvironment } from './environmentTypes';

const cloud = (enabled = true): HostCloudLayer => ({
  enabled, coverage: 0.4, density: 0.6, altitude: 1800, thickness: 700,
  scale: 1, detail: 0.5, softness: 0.4, windX: 1, windY: 0,
  windSpeed: 8, lightingStrength: 1, silverLining: 0.3,
});

const environment = (): HostWorldEnvironment => ({
  entity: { index: 7, generation: 1 }, enabled: true, skyVisible: true, affectLighting: true,
  skySource: 'physicalAtmosphere', solidColor: { x: 0.08, y: 0.13, z: 0.22 }, hdriPath: '',
  hdriRotationDegrees: 0, radianceIntensity: 1, planetRadius: 6360, atmosphereRadius: 6420,
  rayleighStrength: 1, mieStrength: 0.35, ozoneStrength: 1,
  atmosphereTint: { x: 1, y: 1, z: 1 }, groundAlbedo: { x: 0.3, y: 0.3, z: 0.3 },
  mieAnisotropy: 0.8, rayleighScaleHeight: 8, mieScaleHeight: 1.2, multiScatteringFactor: 1,
  exposure: 1, sunDiskSize: 0.01, sunDiskIntensity: 1, sunMode: 'geographic', timeMode: 'fixed',
  latitudeDegrees: 47, longitudeDegrees: 8, northOffsetDegrees: 0, year: 2026, month: 7, day: 15,
  localTimeHours: 10.5, utcOffsetHours: 2, playing: false, loopDay: true, timeScale: 60,
  automaticSunLight: true, sunIntensityMultiplier: 1, sunTemperatureMultiplier: 1,
  moonEnabled: true, automaticMoonPhase: true, moonPhase: 0.5, moonIntensity: 0.2,
  moonAngularRadiusDegrees: 0.26, starsEnabled: true, starDensity: 0.5, starIntensity: 1,
  starTwinkle: 0.1, cloudsEnabled: true, cloudShadows: true, cumulus: cloud(), cirrus: cloud(),
  fogEnabled: true, fogColor: { x: 0.5, y: 0.6, z: 0.7 }, fogDensity: 0.01,
  fogHeightFalloff: 0.2, fogStartDistance: 10, fogMaxOpacity: 0.9, fogSunScattering: 0.5,
  lightingEnabled: true, lightingSource: 'followSky', lightingColor: { x: 0.18, y: 0.23, z: 0.29 },
  diffuseIntensity: 1, specularIntensity: 1,
});

afterEach(cleanup);

describe('schema-driven WorldEnvironmentInspector', () => {
  it('reuses component cards, conditional fields, search, and virtual cloud components', async () => {
    const onChange = vi.fn();
    render(<WorldEnvironmentInspector assets={[]} environment={environment()} onChange={onChange}
      onHdri={vi.fn()} onPreset={vi.fn()} thumbnailProvider={vi.fn()} />);

    expect(screen.getByLabelText('Collapse General')).toBeInTheDocument();
    expect(screen.getByLabelText('Expand Atmosphere')).toBeInTheDocument();
    expect(screen.getByLabelText('Collapse Cumulus Cloud Layer')).toBeInTheDocument();
    expect(screen.queryByLabelText('Choose HDRI Texture asset')).not.toBeInTheDocument();

    await userEvent.selectOptions(screen.getByLabelText('Sky Source'), 'hdri');
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ skySource: 'hdri' }));
    expect(screen.getByLabelText('Choose HDRI Texture asset')).toBeInTheDocument();

    await userEvent.type(screen.getByLabelText('Search world settings'), 'fog');
    expect(screen.getByLabelText('Collapse Fog')).toBeInTheDocument();
    expect(screen.queryByLabelText('Collapse General')).not.toBeInTheDocument();
  });

  it('shows square texture previews and assigns an HDRI through the asset picker', async () => {
    const value = { ...environment(), skySource: 'hdri' as const, hdriPath: 'environments/day.hdr' };
    const onHdri = vi.fn();
    const thumbnailProvider = vi.fn().mockResolvedValue('data:image/bmp;base64,QkFAKE');
    render(<WorldEnvironmentInspector environment={value} onChange={vi.fn()} onHdri={onHdri} onPreset={vi.fn()}
      thumbnailProvider={thumbnailProvider} assets={[
        { id: 'day', name: 'Day HDRI', path: 'environments/day.hdr', kind: 'texture', status: 'ready' },
        { id: 'night', name: 'Night HDRI', path: 'environments/night.hdr', kind: 'texture', status: 'ready' },
        { id: 'albedo', name: 'Rock Albedo', path: 'textures/rock.jpg', kind: 'texture', status: 'ready' },
      ]} />);

    expect(document.querySelector('.asset-reference-control .asset-thumbnail')).toBeInTheDocument();
    await userEvent.click(screen.getByLabelText('Choose HDRI Texture asset'));
    expect(screen.getByRole('dialog', { name: 'HDRI Texture asset picker' })).toBeInTheDocument();
    expect(screen.getByLabelText('Select Day HDRI')).toBeInTheDocument();
    expect(screen.getByLabelText('Select Night HDRI')).toBeInTheDocument();
    expect(screen.queryByLabelText('Select Rock Albedo')).not.toBeInTheDocument();
    await waitFor(() => expect(thumbnailProvider).toHaveBeenCalled());

    await userEvent.click(screen.getByLabelText('Select Night HDRI'));
    expect(onHdri).toHaveBeenCalledWith('environments/night.hdr');
    expect(screen.getByText('Night HDRI')).toBeInTheDocument();
  });
});
