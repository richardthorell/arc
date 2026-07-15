import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';

import { UiButton } from '../ui';
import type { HostCloudLayer, HostWorldEnvironment } from './environmentTypes';

export type WorldEnvironmentInspectorProps = {
  environment: HostWorldEnvironment;
  onChange: (environment: HostWorldEnvironment) => void;
  onPreset: (preset: string) => void;
  onHdri: (path: string) => void;
};

export function WorldEnvironmentInspector({ environment, onChange, onPreset, onHdri }: WorldEnvironmentInspectorProps) {
  const [hdriPath, setHdriPath] = useState(environment.hdriPath);
  useEffect(() => setHdriPath(environment.hdriPath), [environment.hdriPath]);
  const patch = <K extends keyof HostWorldEnvironment>(key: K, value: HostWorldEnvironment[K]) =>
    onChange({ ...environment, [key]: value });

  return (
    <div className="environment-inspector">
      <section className="environment-presets">
        {[
          ['clearDay', 'Clear Day'], ['alpineLateMorning', 'Alpine'], ['goldenHour', 'Golden Hour'],
          ['overcast', 'Overcast'], ['night', 'Night'], ['indoorNeutral', 'Indoor'],
        ].map(([id, label]) => <UiButton key={id} onClick={() => onPreset(id)} variant="toolbar">{label}</UiButton>)}
      </section>

      <EnvironmentSection title="General">
        <EnvironmentToggle label="Enabled" value={environment.enabled} onChange={(value) => patch('enabled', value)} />
        <EnvironmentToggle label="Sky Visible" value={environment.skyVisible} onChange={(value) => patch('skyVisible', value)} />
        <EnvironmentToggle label="Affect Lighting" value={environment.affectLighting} onChange={(value) => patch('affectLighting', value)} />
      </EnvironmentSection>

      <EnvironmentSection title="Sky Source">
        <EnvironmentSelect label="Source" value={environment.skySource} options={[
          ['physicalAtmosphere', 'Physical Atmosphere'], ['hdri', 'HDRI'], ['solidColor', 'Solid Color'],
        ]} onChange={(value) => patch('skySource', value as HostWorldEnvironment['skySource'])} />
        <EnvironmentNumber label="Radiance" min={0} step={0.05} value={environment.radianceIntensity} onChange={(value) => patch('radianceIntensity', value)} />
        <EnvironmentNumber label="Rotation" step={1} value={environment.hdriRotationDegrees} onChange={(value) => patch('hdriRotationDegrees', value)} />
        <label className="environment-path"><span>HDRI</span><input onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); const path = event.dataTransfer.getData('application/x-arc-environment'); if (path) { setHdriPath(path); onHdri(path); } }} onChange={(event) => setHdriPath(event.target.value)} placeholder="environments/studio.hdr" value={hdriPath} /></label>
        <UiButton onClick={() => onHdri(hdriPath)} variant="toolbar">Load HDRI</UiButton>
      </EnvironmentSection>

      <EnvironmentSection title="Sun & Time">
        <EnvironmentSelect label="Sun Position" value={environment.sunMode} options={[
          ['manualLight', 'Manual Light'], ['geographic', 'Geographic'],
        ]} onChange={(value) => patch('sunMode', value as HostWorldEnvironment['sunMode'])} />
        <EnvironmentSelect label="Clock" value={environment.timeMode} options={[
          ['fixed', 'Fixed'], ['simulated', 'Simulated'], ['systemClock', 'System Clock'],
        ]} onChange={(value) => patch('timeMode', value as HostWorldEnvironment['timeMode'])} />
        <EnvironmentToggle label="Play" value={environment.playing} onChange={(value) => patch('playing', value)} />
        <EnvironmentNumber label="Time of Day" min={0} max={23.999} step={0.05} value={environment.localTimeHours} onChange={(value) => patch('localTimeHours', value)} />
        <EnvironmentNumber label="Time Scale" min={0} step={1} value={environment.timeScale} onChange={(value) => patch('timeScale', value)} />
        <EnvironmentNumber label="Latitude" min={-90} max={90} step={0.1} value={environment.latitudeDegrees} onChange={(value) => patch('latitudeDegrees', value)} />
        <EnvironmentNumber label="Longitude" min={-180} max={180} step={0.1} value={environment.longitudeDegrees} onChange={(value) => patch('longitudeDegrees', value)} />
        <EnvironmentNumber label="UTC Offset" min={-14} max={14} step={0.5} value={environment.utcOffsetHours} onChange={(value) => patch('utcOffsetHours', value)} />
        <EnvironmentNumber label="North Offset" step={1} value={environment.northOffsetDegrees} onChange={(value) => patch('northOffsetDegrees', value)} />
        <EnvironmentToggle label="Loop Day" value={environment.loopDay} onChange={(value) => patch('loopDay', value)} />
        <EnvironmentToggle label="Automatic Sun" value={environment.automaticSunLight} onChange={(value) => patch('automaticSunLight', value)} />
        <EnvironmentNumber label="Sun Intensity" min={0} step={0.05} value={environment.sunIntensityMultiplier} onChange={(value) => patch('sunIntensityMultiplier', value)} />
        <EnvironmentNumber label="Sun Temperature" min={0.1} step={0.05} value={environment.sunTemperatureMultiplier} onChange={(value) => patch('sunTemperatureMultiplier', value)} />
        <div className="environment-date">
          <EnvironmentNumber label="Year" min={1} max={9999} step={1} value={environment.year} onChange={(value) => patch('year', Math.trunc(value))} />
          <EnvironmentNumber label="Month" min={1} max={12} step={1} value={environment.month} onChange={(value) => patch('month', Math.trunc(value))} />
          <EnvironmentNumber label="Day" min={1} max={31} step={1} value={environment.day} onChange={(value) => patch('day', Math.trunc(value))} />
        </div>
      </EnvironmentSection>

      <EnvironmentSection title="Atmosphere" advanced>
        <EnvironmentNumber label="Rayleigh" min={0} step={0.02} value={environment.rayleighStrength} onChange={(value) => patch('rayleighStrength', value)} />
        <EnvironmentNumber label="Mie / Haze" min={0} step={0.02} value={environment.mieStrength} onChange={(value) => patch('mieStrength', value)} />
        <EnvironmentNumber label="Ozone" min={0} step={0.02} value={environment.ozoneStrength} onChange={(value) => patch('ozoneStrength', value)} />
        <EnvironmentNumber label="Mie Anisotropy" min={-0.98} max={0.98} step={0.01} value={environment.mieAnisotropy} onChange={(value) => patch('mieAnisotropy', value)} />
        <EnvironmentNumber label="Exposure" min={0} step={0.05} value={environment.exposure} onChange={(value) => patch('exposure', value)} />
        <EnvironmentNumber label="Rayleigh Height km" min={0.01} step={0.1} value={environment.rayleighScaleHeight} onChange={(value) => patch('rayleighScaleHeight', value)} />
        <EnvironmentNumber label="Mie Height km" min={0.01} step={0.1} value={environment.mieScaleHeight} onChange={(value) => patch('mieScaleHeight', value)} />
        <EnvironmentNumber label="Multi Scattering" min={0} step={0.05} value={environment.multiScatteringFactor} onChange={(value) => patch('multiScatteringFactor', value)} />
        <EnvironmentNumber label="Sun Disk Size" min={0} step={0.001} value={environment.sunDiskSize} onChange={(value) => patch('sunDiskSize', value)} />
        <EnvironmentNumber label="Sun Disk Power" min={0} step={0.05} value={environment.sunDiskIntensity} onChange={(value) => patch('sunDiskIntensity', value)} />
        <EnvironmentNumber label="Planet Radius km" min={1} step={1} value={environment.planetRadius} onChange={(value) => patch('planetRadius', value)} />
        <EnvironmentNumber label="Atmosphere Radius km" min={1} step={1} value={environment.atmosphereRadius} onChange={(value) => patch('atmosphereRadius', value)} />
      </EnvironmentSection>

      <EnvironmentSection title="Night Sky">
        <EnvironmentToggle label="Moon" value={environment.moonEnabled} onChange={(value) => patch('moonEnabled', value)} />
        <EnvironmentToggle label="Automatic Phase" value={environment.automaticMoonPhase} onChange={(value) => patch('automaticMoonPhase', value)} />
        <EnvironmentNumber label="Moon Phase" min={0} max={1} step={0.01} value={environment.moonPhase} onChange={(value) => patch('moonPhase', value)} />
        <EnvironmentNumber label="Moon Brightness" min={0} step={0.02} value={environment.moonIntensity} onChange={(value) => patch('moonIntensity', value)} />
        <EnvironmentNumber label="Moon Angular Radius" min={0.01} step={0.01} value={environment.moonAngularRadiusDegrees} onChange={(value) => patch('moonAngularRadiusDegrees', value)} />
        <EnvironmentToggle label="Stars" value={environment.starsEnabled} onChange={(value) => patch('starsEnabled', value)} />
        <EnvironmentNumber label="Star Density" min={0} max={1} step={0.01} value={environment.starDensity} onChange={(value) => patch('starDensity', value)} />
        <EnvironmentNumber label="Star Intensity" min={0} step={0.05} value={environment.starIntensity} onChange={(value) => patch('starIntensity', value)} />
        <EnvironmentNumber label="Star Twinkle" min={0} max={1} step={0.01} value={environment.starTwinkle} onChange={(value) => patch('starTwinkle', value)} />
      </EnvironmentSection>

      <EnvironmentSection title="Clouds">
        <EnvironmentToggle label="Enabled" value={environment.cloudsEnabled} onChange={(value) => patch('cloudsEnabled', value)} />
        <EnvironmentToggle label="Cloud Shadows" value={environment.cloudShadows} onChange={(value) => patch('cloudShadows', value)} />
        <CloudLayerEditor label="Cumulus" layer={environment.cumulus} onChange={(value) => patch('cumulus', value)} />
        <CloudLayerEditor label="Cirrus" layer={environment.cirrus} onChange={(value) => patch('cirrus', value)} />
      </EnvironmentSection>

      <EnvironmentSection title="Fog">
        <EnvironmentToggle label="Enabled" value={environment.fogEnabled} onChange={(value) => patch('fogEnabled', value)} />
        <EnvironmentNumber label="Density" min={0} step={0.001} value={environment.fogDensity} onChange={(value) => patch('fogDensity', value)} />
        <EnvironmentNumber label="Height Falloff" min={0} step={0.01} value={environment.fogHeightFalloff} onChange={(value) => patch('fogHeightFalloff', value)} />
        <EnvironmentNumber label="Start Distance" min={0} step={1} value={environment.fogStartDistance} onChange={(value) => patch('fogStartDistance', value)} />
        <EnvironmentNumber label="Max Opacity" min={0} max={1} step={0.01} value={environment.fogMaxOpacity} onChange={(value) => patch('fogMaxOpacity', value)} />
      </EnvironmentSection>

      <EnvironmentSection title="Environment Lighting">
        <EnvironmentToggle label="Enabled" value={environment.lightingEnabled} onChange={(value) => patch('lightingEnabled', value)} />
        <EnvironmentSelect label="Source" value={environment.lightingSource} options={[
          ['followSky', 'Follow Sky'], ['hdri', 'HDRI'], ['constantColor', 'Constant Color'],
        ]} onChange={(value) => patch('lightingSource', value as HostWorldEnvironment['lightingSource'])} />
        <EnvironmentNumber label="Diffuse" min={0} step={0.05} value={environment.diffuseIntensity} onChange={(value) => patch('diffuseIntensity', value)} />
        <EnvironmentNumber label="Specular" min={0} step={0.05} value={environment.specularIntensity} onChange={(value) => patch('specularIntensity', value)} />
      </EnvironmentSection>
    </div>
  );
}

function EnvironmentSection({ title, children, advanced = false }: { title: string; children: ReactNode; advanced?: boolean }) {
  return <details className="environment-section" open={!advanced}><summary><ChevronDown size={14} />{title}{advanced && <small>Advanced</small>}</summary><div>{children}</div></details>;
}

function EnvironmentToggle({ label, value, onChange }: { label: string; value: boolean; onChange: (value: boolean) => void }) {
  return <label className="environment-control"><span>{label}</span><input checked={value} onChange={(event) => onChange(event.target.checked)} type="checkbox" /></label>;
}

function EnvironmentNumber({ label, value, onChange, min, max, step }: { label: string; value: number; onChange: (value: number) => void; min?: number; max?: number; step?: number }) {
  return <label className="environment-control"><span>{label}</span><input min={min} max={max} step={step} type="number" value={Number.isFinite(value) ? value : 0} onChange={(event) => { const next = Number(event.target.value); if (Number.isFinite(next)) onChange(next); }} /></label>;
}

function EnvironmentSelect({ label, value, options, onChange }: { label: string; value: string; options: string[][]; onChange: (value: string) => void }) {
  return <label className="environment-control"><span>{label}</span><select value={value} onChange={(event) => onChange(event.target.value)}>{options.map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></label>;
}

function CloudLayerEditor({ label, layer, onChange }: { label: string; layer: HostCloudLayer; onChange: (layer: HostCloudLayer) => void }) {
  const patch = <K extends keyof HostCloudLayer>(key: K, value: HostCloudLayer[K]) => onChange({ ...layer, [key]: value });
  return <fieldset className="cloud-layer"><legend>{label}</legend><EnvironmentToggle label="Enabled" value={layer.enabled} onChange={(value) => patch('enabled', value)} /><EnvironmentNumber label="Coverage" min={0} max={1} step={0.01} value={layer.coverage} onChange={(value) => patch('coverage', value)} /><EnvironmentNumber label="Density" min={0} max={1} step={0.01} value={layer.density} onChange={(value) => patch('density', value)} /><EnvironmentNumber label="Altitude" min={0} step={50} value={layer.altitude} onChange={(value) => patch('altitude', value)} /><EnvironmentNumber label="Thickness" min={0} step={10} value={layer.thickness} onChange={(value) => patch('thickness', value)} /><EnvironmentNumber label="Wind Speed" min={0} step={0.5} value={layer.windSpeed} onChange={(value) => patch('windSpeed', value)} /><EnvironmentNumber label="Silver Lining" min={0} max={1} step={0.01} value={layer.silverLining} onChange={(value) => patch('silverLining', value)} /></fieldset>;
}
