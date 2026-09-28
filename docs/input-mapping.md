# Input mapping configuration

ARC keeps gameplay intent separate from physical input through semantic actions and axes. Project input configuration is loaded from `arc.project.json` by the project module and applied to an `arc::input::input_player` at runtime.

## Configuration shape

Input configuration lives under the top-level `input` object:

```json
{
  "input": {
    "actions": [
      { "name": "Jump", "trigger": "pressed" },
      { "name": "Pause", "trigger": "released" }
    ],
    "axes": [
      { "name": "Move", "dimensions": 2 },
      { "name": "Look", "dimensions": 2 },
      { "name": "Zoom", "dimensions": 1 }
    ],
    "bindings": [
      { "action": "Jump", "control": "key.space" },
      { "action": "Pause", "control": "key.escape" },
      { "axis": "Move", "control": "key.w", "contribution": [0.0, 1.0] },
      { "axis": "Move", "control": "key.s", "contribution": [0.0, -1.0] },
      { "axis": "Move", "control": "key.a", "contribution": [-1.0, 0.0] },
      { "axis": "Move", "control": "key.d", "contribution": [1.0, 0.0] },
      { "axis": "Look", "control": "mouse.delta" },
      { "axis": "Zoom", "control": "mouse.wheel", "processors": [{ "type": "scale", "value": 0.25 }] }
    ]
  }
}
```

`actions` describe discrete semantic events. `trigger` accepts `pressed`, `released`, `down`, or `up`.

`axes` describe continuous semantic values. `dimensions` must be `1` or `2` and must match the binding contribution when one is supplied.

Each binding targets exactly one declared `action` or `axis`. Bindings are resolved when the configuration is applied to an input player, so gameplay code consumes semantic names rather than physical controls.

## Supported controls

Keyboard controls use `key.<name>`. The current project parser recognizes:

- `key.space`, `key.escape`, `key.enter`, `key.tab`, `key.backspace`
- `key.left`, `key.right`, `key.up`, `key.down`
- `key.a` through `key.z`
- `key.0` through `key.9`

Mouse controls use:

- `mouse.left`, `mouse.right`, `mouse.middle`, `mouse.x1`, `mouse.x2`
- `mouse.position` for absolute pointer position
- `mouse.delta` for per-frame pointer motion
- `mouse.wheel` for wheel input

Unknown control names are rejected instead of silently producing an unusable binding.

## Axis contributions

A scalar axis may use a numeric contribution:

```json
{ "axis": "Throttle", "control": "key.w", "contribution": 1.0 }
```

A two-dimensional axis may use a two-element array:

```json
{ "axis": "Move", "control": "key.a", "contribution": [-1.0, 0.0] }
```

When no contribution is specified, the physical control value is used directly. Contributions are useful for combining digital controls into a semantic scalar or vector axis without exposing key identities to gameplay systems.

## Processors

Bindings may include a `processors` array. Processors are evaluated in declaration order.

```json
{
  "axis": "LookX",
  "control": "mouse.delta",
  "processors": [
    { "type": "deadzone", "value": 0.05 },
    { "type": "scale", "value": 0.5 },
    { "type": "invert" }
  ]
}
```

Supported processor types are:

- `deadzone` with a finite `value` in the inclusive range `0.0` to `1.0`
- `scale` with a finite numeric `value`
- `invert`, which takes no value

Malformed processor definitions fail configuration loading with a descriptive error rather than being ignored.

## Runtime usage

The project module owns parsing and validation. Runtime input code receives the already validated configuration and applies it through `arc::project::apply_input_config(...)`:

```cpp
arc::input::input_player player;
std::string error;

if (!arc::project::apply_input_config(project.input, player, error)) {
    // Report the project configuration error.
}
```

Gameplay and editor systems should query the resulting semantic actions and axes through `input_player`; they should not duplicate project-file parsing or couple semantic behavior to platform key codes.

## Validation expectations

Configuration loading is intentionally strict. Treat a load failure as a project-authoring error and surface the returned message. In particular:

- semantic names must be non-empty and unique within their category;
- bindings must reference declared actions or axes;
- a binding must target exactly one semantic input;
- controls must use a supported canonical name;
- processor arguments must be finite and within their documented ranges;
- axis contribution shape must match the declared axis dimensions.

This keeps invalid mappings from becoming platform-dependent runtime behavior and gives tooling a stable schema to validate against.