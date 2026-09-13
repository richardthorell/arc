from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    if old not in text:
        raise SystemExit(f"anchor not found in {path}: {old[:100]!r}")
    file.write_text(text.replace(old, new, 1))


# Native viewport protocol: carry a configurable base grid color with render options.
path = Path('editor/native/inc/arc/editor/host_protocol_base.h')
text = path.read_text()
anchor = '    bool grid{true};\n    bool skeletons{};'
if text.count(anchor) != 2:
    raise SystemExit(f'expected two viewport grid fields, found {text.count(anchor)}')
text = text.replace(
    anchor,
    '    bool grid{true};\n'
    '    host_vec3 grid_color{0.2f, 0.21568628f, 0.23921569f};\n'
    '    bool skeletons{};',
)
path.write_text(text)

replace_once(
    'editor/native/src/host_protocol_base.inc',
    '        bool_value(payload, "grid", command.grid);\n        bool_value(payload, "skeletons", command.skeletons);',
    '        bool_value(payload, "grid", command.grid);\n'
    '        array3_value(payload, "gridColor", command.grid_color);\n'
    '        bool_value(payload, "skeletons", command.skeletons);',
)

replace_once(
    'editor/native/src/arc_host_base.inc',
    '                state_->viewport_options.grid = payload.grid;\n'
    '                state_->viewport_options.skeletons = payload.skeletons;',
    '                state_->viewport_options.grid = payload.grid;\n'
    '                state_->viewport_options.grid_color = payload.grid_color;\n'
    '                state_->viewport_options.skeletons = payload.skeletons;',
)
replace_once(
    'editor/native/src/arc_host_base.inc',
    '            append_editor_grid_overlay(debug_overlay, *camera, *camera_transform, request.height);',
    '            append_editor_grid_overlay(debug_overlay, *camera, *camera_transform, request.height,\n'
    '                                       {request.grid_color.x, request.grid_color.y, request.grid_color.z});',
)

replace_once(
    'editor/native/inc/arc/editor/editor_gizmo.h',
    'void append_editor_grid_overlay(render::debug_overlay_stream& stream, const scene::camera_component& camera,\n'
    '                                const scene::transform_component& camera_transform, std::uint32_t viewport_height);',
    'void append_editor_grid_overlay(render::debug_overlay_stream& stream, const scene::camera_component& camera,\n'
    '                                const scene::transform_component& camera_transform, std::uint32_t viewport_height,\n'
    '                                const math::vector3f& grid_color = {0.2f, 0.21568628f, 0.23921569f});',
)

# Restyle the adaptive XZ grid while retaining the existing spacing/extent behavior.
gizmo = Path('editor/native/src/editor_gizmo.cpp')
text = gizmo.read_text()
old_sig = '''void append_editor_grid_overlay(render::debug_overlay_stream& stream, const scene::camera_component& camera,
                                const scene::transform_component& camera_transform, std::uint32_t viewport_height)
'''
new_sig = '''void append_editor_grid_overlay(render::debug_overlay_stream& stream, const scene::camera_component& camera,
                                const scene::transform_component& camera_transform, std::uint32_t viewport_height,
                                const math::vector3f& grid_color)
'''
if old_sig not in text:
    raise SystemExit('grid function signature anchor not found')
text = text.replace(old_sig, new_sig, 1)
old_colors = '''    constexpr math::vector4f minor_color{1.0f, 1.0f, 1.0f, 0.20f};
    constexpr math::vector4f major_color{1.0f, 1.0f, 1.0f, 0.38f};
    constexpr math::vector4f axis_color{1.0f, 1.0f, 1.0f, 0.55f};
    const auto grid_color = [&](float coordinate)
    {
        if (std::abs(coordinate) <= spacing * 0.25f) return axis_color;
        const auto world_line = static_cast<long long>(std::llround(coordinate / spacing));
        return world_line % grid_major_interval == 0 ? major_color : minor_color;
    };
'''
new_colors = '''    const auto color = [&](float multiplier, float alpha)
    {
        return math::vector4f{std::clamp(grid_color[0] * multiplier, 0.0f, 1.0f),
                              std::clamp(grid_color[1] * multiplier, 0.0f, 1.0f),
                              std::clamp(grid_color[2] * multiplier, 0.0f, 1.0f), alpha};
    };
    const auto minor_color = color(1.0f, 0.45f);
    const auto major_color = color(1.45f, 0.60f);
    constexpr math::vector4f x_axis_color{0.7882353f, 0.3372549f, 0.2980392f, 0.85f};
    constexpr math::vector4f z_axis_color{0.2980392f, 0.4980392f, 0.8196079f, 0.85f};
    const auto regular_grid_color = [&](float coordinate)
    {
        const auto world_line = static_cast<long long>(std::llround(coordinate / spacing));
        return world_line % grid_major_interval == 0 ? major_color : minor_color;
    };
'''
if old_colors not in text:
    raise SystemExit('grid color block anchor not found')
text = text.replace(old_colors, new_colors, 1)
text = text.replace(
    '.color = grid_color(z),\n                                .depth = render::debug_overlay_depth_mode::tested});',
    '.color = std::abs(z) <= spacing * 0.25f ? x_axis_color : regular_grid_color(z),\n'
    '                                .depth = render::debug_overlay_depth_mode::tested});',
    1,
)
text = text.replace(
    '.color = grid_color(x),\n                                .depth = render::debug_overlay_depth_mode::tested});',
    '.color = std::abs(x) <= spacing * 0.25f ? z_axis_color : regular_grid_color(x),\n'
    '                                .depth = render::debug_overlay_depth_mode::tested});',
    1,
)
gizmo.write_text(text)

# Generic settings metadata supports color inputs while retaining string persistence.
replace_once(
    'editor/src/common/editorWorkflowTypes.ts',
    "  type: 'boolean' | 'number' | 'string' | 'enum';\n  defaultValue: boolean | number | string;",
    "  type: 'boolean' | 'number' | 'string' | 'enum';\n  format?: 'color';\n  defaultValue: boolean | number | string;",
)

settings = Path('editor/src/main/settingsService.ts')
text = settings.read_text()
anchor = '''  {
    key: 'renderer.qualityTier',
'''
entry = '''  {
    key: 'renderer.gridColor',
    section: 'Renderer',
    label: 'Grid Color',
    description: 'Base color used by the editor viewport grid.',
    type: 'string',
    format: 'color',
    defaultValue: '#33373D',
    scopes: ['user', 'project'],
  },
'''
if anchor not in text:
    raise SystemExit('settings schema renderer anchor not found')
text = text.replace(anchor, entry + anchor, 1)
old = "  if (descriptor.type === 'string' && typeof value !== 'string') throw new Error(`${descriptor.key} must be a string`);\n"
new = old + "  if (descriptor.format === 'color' && (typeof value !== 'string' || !/^#[0-9a-fA-F]{6}$/.test(value)))\n    throw new Error(`${descriptor.key} must be a #RRGGBB color`);\n"
if old not in text:
    raise SystemExit('settings validation anchor not found')
settings.write_text(text.replace(old, new, 1))

# Settings UI: native color control and a lightweight in-process notification for live viewport updates.
dialog = Path('editor/src/renderer/src/settings/SettingsDialog.tsx')
text = dialog.read_text()
old = '''      const next = await window.arc.settings.update(scope, { [key]: value }, snapshot.revision);
      if (next) setSnapshot(next);
      setMessage(`${key} updated in ${scope} settings`);
'''
new = '''      const next = await window.arc.settings.update(scope, { [key]: value }, snapshot.revision);
      if (next) {
        setSnapshot(next);
        window.dispatchEvent(new CustomEvent('arc-editor-settings-changed', { detail: next }));
      }
      setMessage(`${key} updated in ${scope} settings`);
'''
if old not in text:
    raise SystemExit('settings update anchor not found')
text = text.replace(old, new, 1)
anchor = '''    if (descriptor.type === 'enum')
'''
color_editor = '''    if (descriptor.format === 'color' && typeof value === 'string')
      return (
        <input
          aria-label={descriptor.label}
          className="settings-color-control"
          disabled={disabled}
          onChange={(event) => void update(key, event.target.value.toUpperCase())}
          type="color"
          value={value}
        />
      );
'''
if anchor not in text:
    raise SystemExit('settings editor anchor not found')
text = text.replace(anchor, color_editor + anchor, 1)
dialog.write_text(text)

css = Path('editor/src/renderer/src/settings/SettingsDialog.css')
css_text = css.read_text()
if '.settings-color-control' not in css_text:
    css_text += '''

.settings-color-control {
  width: 44px;
  height: 28px;
  padding: 2px;
  border: 1px solid var(--arc-color-border);
  border-radius: var(--arc-radius-sm);
  background: var(--arc-color-bg-control);
  cursor: pointer;
}

.settings-color-control:disabled {
  cursor: default;
  opacity: 0.55;
}
'''
css.write_text(css_text)

# Viewport reads the persisted setting, applies it after attach, and reacts live to Settings edits.
viewport = Path('editor/src/renderer/src/viewport/ViewportPanel.tsx')
text = viewport.read_text()
anchor = "const formatNumber = (value: number) => Math.max(0, value).toLocaleString();\n"
helpers = '''const defaultGridColor = '#33373D';
const gridColorValue = (value: string) => {
  const match = /^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$/.exec(value);
  if (!match) return [0.2, 0.21568628, 0.23921569] as const;
  return [Number.parseInt(match[1], 16) / 255, Number.parseInt(match[2], 16) / 255, Number.parseInt(match[3], 16) / 255] as const;
};

'''
if anchor not in text:
    raise SystemExit('viewport helper anchor not found')
text = text.replace(anchor, helpers + anchor, 1)
anchor = '''  const [viewportStats, setViewportStats] = useState<ViewportStats>(() => fallbackStats(project));
  const [localGridVisible, setLocalGridVisible] = useState(true);
'''
replacement = '''  const [viewportStats, setViewportStats] = useState<ViewportStats>(() => fallbackStats(project));
  const renderOptionsRef = useRef<ViewportRenderOptions>(defaultRenderOptions);
  const gridColorRef = useRef(defaultGridColor);
  const [localGridVisible, setLocalGridVisible] = useState(true);
'''
if anchor not in text:
    raise SystemExit('viewport state anchor not found')
text = text.replace(anchor, replacement, 1)

# Place sender before attachViewport.
anchor = '''  const attachViewport = useCallback(async () => {
'''
sender = '''  const sendGridColor = useCallback(async (value: string) => {
    if (!viewportActive || !viewportAttachedRef.current) return;
    const [red, green, blue] = gridColorValue(value);
    const response = (await window.arc.host.command('viewport.setRenderOptions', {
      viewportId,
      ...renderOptionsRef.current,
      gridColor: [red, green, blue],
    })) as ViewportCommandResponse;
    if (response?.succeeded === false) throw new Error(response.error || 'Could not update viewport grid color');
  }, [viewportActive, viewportId]);

'''
if anchor not in text:
    raise SystemExit('viewport attach anchor not found')
text = text.replace(anchor, sender + anchor, 1)

old = '''      viewportAttachedRef.current = true;
      lastViewportBoundsRef.current = boundsKey(bounds);
'''
new = '''      viewportAttachedRef.current = true;
      lastViewportBoundsRef.current = boundsKey(bounds);
      void sendGridColor(gridColorRef.current).catch((error) => {
        setViewportError(error instanceof Error ? error.message : String(error));
      });
'''
if old not in text:
    raise SystemExit('viewport attach success anchor not found')
text = text.replace(old, new, 1)
# Update useCallback dependency to include sender without relying on exact full list.
text = text.replace(
    '  }, [recordSharedFailure, streamedAvailable, viewportActive, viewportBounds]);',
    '  }, [recordSharedFailure, sendGridColor, streamedAvailable, viewportActive, viewportBounds]);',
    1,
)

# Apply initial/persisted setting and live dialog changes.
anchor = '''  useEffect(() => {
    if (!playSessionActive) setPlayInputCaptured(false);
  }, [playSessionActive]);
'''
settings_effect = '''  useEffect(() => {
    let cancelled = false;
    const applySnapshot = (next: import('../../../common/editorWorkflowTypes').EditorSettingsSnapshot | null) => {
      if (cancelled) return;
      const value = next?.values['renderer.gridColor'];
      if (typeof value !== 'string') return;
      gridColorRef.current = value;
      void sendGridColor(value).catch((error) => {
        if (!cancelled) setViewportError(error instanceof Error ? error.message : String(error));
      });
    };
    void window.arc.settings.snapshot().then(applySnapshot);
    const onSettingsChanged = (event: Event) =>
      applySnapshot((event as CustomEvent<import('../../../common/editorWorkflowTypes').EditorSettingsSnapshot>).detail);
    window.addEventListener('arc-editor-settings-changed', onSettingsChanged);
    return () => {
      cancelled = true;
      window.removeEventListener('arc-editor-settings-changed', onSettingsChanged);
    };
  }, [sendGridColor]);

'''
if anchor not in text:
    raise SystemExit('viewport play-session effect anchor not found')
text = text.replace(anchor, settings_effect + anchor, 1)

# Keep a current render-options copy from host polling.
old = '''        if (!cancelled && response?.succeeded && response.payload) {
          setViewportStats(response.payload);
          if (typeof response.payload.renderOptions?.grid === 'boolean')
'''
new = '''        if (!cancelled && response?.succeeded && response.payload) {
          if (response.payload.renderOptions) renderOptionsRef.current = response.payload.renderOptions;
          setViewportStats(response.payload);
          if (typeof response.payload.renderOptions?.grid === 'boolean')
'''
if old not in text:
    raise SystemExit('viewport stats anchor not found')
text = text.replace(old, new, 1)

# Every existing render-options mutation carries the configured grid color too.
old = '''        viewportId,
        ...renderOptions,
        grid: visible,
'''
new = '''        viewportId,
        ...renderOptions,
        grid: visible,
        gridColor: gridColorValue(gridColorRef.current),
'''
if old not in text:
    raise SystemExit('viewport grid toggle payload anchor not found')
text = text.replace(old, new, 1)
old = '''    const next = { ...previous, ...changes, environment: changes.environment ?? previous.environment };
    setViewportStats((current) => ({ ...current, renderOptions: next }));
'''
new = '''    const next = { ...previous, ...changes, environment: changes.environment ?? previous.environment };
    renderOptionsRef.current = next;
    setViewportStats((current) => ({ ...current, renderOptions: next }));
'''
if old not in text:
    raise SystemExit('viewport render update state anchor not found')
text = text.replace(old, new, 1)
old = '''        viewportId,
        ...next,
      })) as ViewportCommandResponse;
'''
new = '''        viewportId,
        ...next,
        gridColor: gridColorValue(gridColorRef.current),
      })) as ViewportCommandResponse;
'''
if old not in text:
    raise SystemExit('viewport render options payload anchor not found')
text = text.replace(old, new, 1)
viewport.write_text(text)

# Settings service regression coverage.
test = Path('editor/src/main/settingsService.test.ts')
text = test.read_text()
anchor = '''  it('rejects unknown, out-of-range, machine-only, stale, and read-only changes', () => {
'''
case = '''  it('persists and validates the viewport grid color', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-settings-'));
    roots.push(root);
    const service = new SettingsService(path.join(root, 'user.json'), () => project(root));
    let snapshot = service.snapshot();

    expect(snapshot.values['renderer.gridColor']).toBe('#33373D');
    snapshot = service.update('user', { 'renderer.gridColor': '#4A5058' }, snapshot.revision);
    expect(snapshot.values['renderer.gridColor']).toBe('#4A5058');
    expect(snapshot.sources['renderer.gridColor']).toBe('user');
    expect(() => service.update('user', { 'renderer.gridColor': 'white' }, snapshot.revision)).toThrow('#RRGGBB');
  });

'''
if anchor not in text:
    raise SystemExit('settings test anchor not found')
test.write_text(text.replace(anchor, case + anchor, 1))

print('Applied editor grid color patch')
