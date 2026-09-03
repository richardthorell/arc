from pathlib import Path

service_path = Path("editor/src/main/projectService.ts")
service = service_path.read_text()
marker = "  async openOrCreateQuickStartProject(destination: string): Promise<ArcProjectOperationResult> {"
start = service.find(marker)
if start < 0:
    raise SystemExit("quick-start method not found")
next_method = service.find("\n  async ", start + len(marker))
if next_method < 0:
    next_method = service.find("\n  create(", start + len(marker))
if next_method < 0:
    raise SystemExit("next ProjectService method not found")
method = service[start:next_method]
open_marker = "      return await this.open(descriptorPath, { upgrade: true });"
if open_marker not in method:
    raise SystemExit("quick-start open return not found")

migration = """      const legacyScenePath = 'Content/Scenes/Startup.arcscene';
      if (descriptorPath && fs.existsSync(descriptorPath)) {
        const rawDescriptor = JSON.parse(fs.readFileSync(descriptorPath, 'utf8')) as Record<string, unknown>;
        const defaultScene = objectValue(rawDescriptor.defaultScene);
        const startupScenes = Array.isArray(rawDescriptor.startupScenes) ? rawDescriptor.startupScenes : [];
        const normalizedPath = (value: unknown): string =>
          typeof value === 'string' ? value.replaceAll('\\\\', '/') : '';
        const referencesLegacyScene =
          normalizedPath(defaultScene.pathHint) === legacyScenePath ||
          startupScenes.some((entry) =>
            normalizedPath(typeof entry === 'string' ? entry : objectValue(entry).pathHint) === legacyScenePath,
          );
        if (referencesLegacyScene) {
          rawDescriptor.defaultScene = null;
          rawDescriptor.startupScenes = [];
          writeJsonAtomic(descriptorPath, rawDescriptor);
          fs.rmSync(path.join(projectRoot, legacyScenePath), { force: true });
          fs.rmSync(path.join(projectRoot, `${legacyScenePath}.arcmeta`), { force: true });
        }
      }

"""
if "const legacyScenePath" not in method:
    method = method.replace(open_marker, migration + open_marker, 1)
service_path.write_text(service[:start] + method + service[next_method:])

descriptor_path = Path("templates/blank-3d/__PROJECT__.arcproject")
descriptor = descriptor_path.read_text()
old = '  "defaultScene": {"guid":"{{SCENE_ASSET_GUID}}","expectedType":"scene","pathHint":"Content/Scenes/Startup.arcscene"},\n  "startupScenes": [{"guid":"{{SCENE_ASSET_GUID}}","expectedType":"scene","pathHint":"Content/Scenes/Startup.arcscene"}],'
new = '  "defaultScene": null,\n  "startupScenes": [],'
if old not in descriptor:
    raise SystemExit("blank-3d startup scene descriptor marker not found")
descriptor_path.write_text(descriptor.replace(old, new, 1))

for candidate in (
    Path("templates/blank-3d/Content/Scenes/Startup.arcscene"),
    Path("templates/blank-3d/Content/Scenes/Startup.arcscene.arcmeta"),
):
    candidate.unlink()

test_path = Path("editor/src/main/projectService.test.ts")
test = test_path.read_text()
old_assert = "    expect(first.project?.descriptor.defaultScene?.pathHint).toBe('Content/Scenes/Startup.arcscene');"
replacement = """    expect(first.project?.descriptor.defaultScene).toBeNull();
    expect(first.project?.descriptor.startupScenes).toEqual([]);
    expect(fs.existsSync(path.join(destination, 'Content/Scenes/Startup.arcscene'))).toBe(false);"""
if old_assert not in test:
    raise SystemExit("quick-start default scene assertion not found")
test = test.replace(old_assert, replacement, 1)
test = test.replace(
    "  it('creates and reuses a Blank 3D quick-start project', async () => {",
    "  it('creates and reuses a Blank 3D quick-start project with the native default scene', async () => {",
    1,
)
test_path.write_text(test)
