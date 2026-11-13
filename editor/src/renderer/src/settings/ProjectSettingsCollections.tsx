import type { ArcProjectAssetReference, ArcProjectDescriptor } from '../../../common/projectTypes';
import type { ProjectSettingsSnapshot } from '../../../common/projectSettingsTypes';
import { AssetPicker, type AssetPickerItem } from '../inspector/AssetPicker';
import { UiButton, UiSelect, UiTextInput } from '../ui';

const options = (values: string[]) => values.map((value) => ({ value, label: value }));
export function ProjectSceneSetting({
  value,
  assets,
  onChange,
  label,
}: {
  value: ArcProjectAssetReference | null;
  assets: AssetPickerItem[];
  onChange: (value: ArcProjectAssetReference | null) => void;
  label: string;
}) {
  const missing = value && !assets.some((asset) => asset.guid === value.guid);
  return (
    <div>
      <AssetPicker
        assets={assets}
        value={value?.guid ?? ''}
        label={label}
        assetKinds={['scene']}
        referenceMode="guid"
        allowEmpty
        onChange={(guid, asset) =>
          onChange(
            guid
              ? { guid, expectedType: 'scene', pathHint: asset?.sourcePath ?? asset?.path ?? value?.pathHint ?? '' }
              : null,
          )
        }
      />
      {missing && (
        <small role="status">Scene unavailable: {value.pathHint || value.guid}. The GUID binding is preserved.</small>
      )}
    </div>
  );
}

export function ProjectSceneList({
  value,
  assets,
  onChange,
}: {
  value: ArcProjectAssetReference[];
  assets: AssetPickerItem[];
  onChange: (value: ArcProjectAssetReference[]) => void;
}) {
  return (
    <div className="project-settings-collection">
      {value.map((scene, index) => (
        <div key={`${scene.guid}-${index}`} className="project-settings-collection-row">
          <ProjectSceneSetting
            label={`Startup scene ${index + 1}`}
            value={scene}
            assets={assets}
            onChange={(next) =>
              onChange(next ? value.map((item, i) => (i === index ? next : item)) : value.filter((_, i) => i !== index))
            }
          />
          <UiButton
            disabled={!index}
            onClick={() => {
              const next = [...value];
              [next[index - 1], next[index]] = [next[index], next[index - 1]];
              onChange(next);
            }}
          >
            Move up
          </UiButton>
          <UiButton onClick={() => onChange(value.filter((_, i) => i !== index))}>Remove scene</UiButton>
        </div>
      ))}
      <ProjectSceneSetting
        label="Add startup scene"
        value={null}
        assets={assets}
        onChange={(scene) => {
          if (scene && !value.some((entry) => entry.guid === scene.guid)) onChange([...value, scene]);
        }}
      />
    </div>
  );
}

export function ProjectTargetList({
  value,
  supported,
  onChange,
}: {
  value: ArcProjectDescriptor['targetPlatforms'];
  supported: string[];
  onChange: (value: ArcProjectDescriptor['targetPlatforms']) => void;
}) {
  return (
    <div className="project-settings-collection">
      {[...new Set([...supported, ...value.map((entry) => entry.id)])].map((id) => (
        <label key={id}>
          <input
            type="checkbox"
            checked={value.find((entry) => entry.id === id)?.enabled ?? false}
            disabled={!supported.includes(id)}
            onChange={(event) =>
              onChange(
                value.some((entry) => entry.id === id)
                  ? value.map((entry) => (entry.id === id ? { ...entry, enabled: event.target.checked } : entry))
                  : [...value, { id, enabled: event.target.checked }],
              )
            }
          />
          {id}
          {!supported.includes(id) && ' (unsupported by this engine)'}
        </label>
      ))}
      <small>
        Enabled targets are project intent, not proof of local packaging or launch support. Disabling retains profiles.
      </small>
    </div>
  );
}

type Profile = ArcProjectDescriptor['cookProfiles'][number];
function targetFields(id: string) {
  const [platform, architecture, renderer] = id.split('-');
  return {
    platform,
    architecture: architecture === 'x64' ? 'x86_64' : architecture,
    renderer: renderer === 'headless' ? 'none' : renderer,
  };
}
export function ProjectProfileList({
  value,
  snapshot,
  targets,
  onChange,
}: {
  value: Profile[];
  snapshot: ProjectSettingsSnapshot;
  targets: ArcProjectDescriptor['targetPlatforms'];
  onChange: (value: Profile[]) => void;
}) {
  const supported = snapshot.capabilities.targets.filter((id) => targets.some((target) => target.id === id));
  const uniqueId = (base: string) => {
    let id = base;
    let suffix = 1;
    while (value.some((entry) => entry.id === id)) id = `${base}-${suffix++}`;
    return id;
  };
  return (
    <div className="project-settings-collection">
      {value.map((profile, index) => {
        const update = (patch: Partial<Profile>) =>
          onChange(value.map((item, i) => (i === index ? { ...item, ...patch } : item)));
        const target = `${profile.platform}-${profile.architecture === 'x86_64' ? 'x64' : profile.architecture}-${profile.renderer === 'none' ? 'headless' : profile.renderer}`;
        return (
          <div key={index} className="project-settings-profile">
            <label>
              Profile ID
              <UiTextInput
                aria-label={`Profile ${index + 1} ID`}
                value={profile.id}
                onChange={(event) => update({ id: event.target.value })}
              />
            </label>
            <label>
              Platform / architecture / renderer
              <UiSelect
                ariaLabel={`Profile ${index + 1} target`}
                value={target}
                options={options([...new Set([target, ...supported])])}
                onValueChange={(id) => update(targetFields(id))}
              />
            </label>
            <label>
              Configuration
              <UiSelect
                ariaLabel={`Profile ${index + 1} configuration`}
                value={profile.configuration}
                options={options([...new Set([profile.configuration, ...snapshot.configurations])])}
                onValueChange={(configuration) => update({ configuration })}
              />
            </label>
            <label>
              Graphics API
              <UiTextInput
                aria-label={`Profile ${index + 1} API`}
                value={profile.api}
                onChange={(event) => update({ api: event.target.value })}
              />
            </label>
            <label>
              Texture quality
              <UiSelect
                ariaLabel={`Profile ${index + 1} texture quality`}
                value={profile.textures.quality}
                options={options(snapshot.capabilities.textureQualities)}
                onValueChange={(quality) => update({ textures: { ...profile.textures, quality } })}
              />
            </label>
            <div>
              Texture outputs
              {snapshot.capabilities.textureOutputs.map((family) => (
                <label key={family}>
                  <input
                    type="checkbox"
                    checked={profile.textures.outputs.includes(family)}
                    onChange={(event) =>
                      update({
                        textures: {
                          ...profile.textures,
                          outputs: event.target.checked
                            ? [...profile.textures.outputs, family]
                            : profile.textures.outputs.filter((entry) => entry !== family),
                        },
                      })
                    }
                  />
                  {family}
                </label>
              ))}
            </div>
            <div>
              <UiButton
                onClick={() =>
                  onChange([...value, { ...structuredClone(profile), id: uniqueId(`${profile.id}-copy`) }])
                }
              >
                Duplicate profile
              </UiButton>
              <UiButton onClick={() => onChange(value.filter((_, i) => i !== index))}>Remove profile</UiButton>
            </div>
          </div>
        );
      })}
      <UiButton
        disabled={!supported.length || value.length >= 64}
        onClick={() =>
          onChange([
            ...value,
            {
              id: uniqueId('profile'),
              ...targetFields(supported[0]),
              api: '1.2',
              configuration: snapshot.configurations.includes('Shipping') ? 'Shipping' : snapshot.configurations[0],
              textures: { outputs: [supported[0].startsWith('android-') ? 'astc' : 'bc'], quality: 'balanced' },
            },
          ])
        }
      >
        Add profile
      </UiButton>
    </div>
  );
}
