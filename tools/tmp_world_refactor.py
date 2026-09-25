from pathlib import Path

path = Path('editor/src/renderer/src/app/Workbench.tsx')
text = path.read_text()

if "const [inspectorTarget, setInspectorTarget]" in text:
    print('Workbench refactor already applied')
    raise SystemExit(0)


def replace(old: str, new: str) -> None:
    global text
    if old not in text:
        raise SystemExit(f'Missing expected Workbench snippet:\n{old}')
    text = text.replace(old, new, 1)


replace("  FolderTree,\n  Lightbulb,", "  FolderTree,\n  Globe2,\n  Lightbulb,")
replace("  Search,\n  Settings,\n  Trash2,", "  Search,\n  Trash2,")
replace("import { LightingPanel } from '../lighting/LightingPanel';\n", "")
replace(
    "  const [worldEnvironment, setWorldEnvironment] = useState<HostWorldEnvironment | null>(null);\n",
    "  const [worldEnvironment, setWorldEnvironment] = useState<HostWorldEnvironment | null>(null);\n"
    "  const [inspectorTarget, setInspectorTarget] = useState<'entity' | 'world'>('entity');\n",
)
replace(
    "        setLastCommand(event.message || event.type);\n        if (event.payload && typeof event.payload === 'object' && 'tool' in event.payload) {",
    "        if (event.type === 'entity.selected' && validHostEntity(event.entity)) setInspectorTarget('entity');\n"
    "        setLastCommand(event.message || event.type);\n"
    "        if (event.payload && typeof event.payload === 'object' && 'tool' in event.payload) {",
)
replace(
    "    if (activeTool !== 'terrain') return;\n    if (selectedSnapshot && !selectedSnapshot.terrain) {",
    "    if (activeTool !== 'terrain') return;\n"
    "    if (inspectorTarget === 'world' || (selectedSnapshot && !selectedSnapshot.terrain)) {",
)
replace(
    "    activeTool,\n    coordinateSpace,\n    refreshTerrainToolState,",
    "    activeTool,\n    coordinateSpace,\n    inspectorTarget,\n    refreshTerrainToolState,",
)
replace(
    "    hasSelection: Boolean(selectedEntityId),",
    "    hasSelection: inspectorTarget === 'entity' && Boolean(selectedEntityId),",
)
replace(
    "        } else if (command === 'entity.duplicate' && selectedSnapshot) {",
    "        } else if (command === 'entity.duplicate' && inspectorTarget === 'entity' && selectedSnapshot) {",
)
replace(
    "        } else if (command === 'entity.delete' && selectedSnapshot) {",
    "        } else if (command === 'entity.delete' && inspectorTarget === 'entity' && selectedSnapshot) {",
)
replace(
    "          if (command === 'viewport.snapToFloor' && selectedSnapshot) {",
    "          if (command === 'viewport.snapToFloor' && inspectorTarget === 'entity' && selectedSnapshot) {",
)
replace(
    "  const selectEntity = async (entityId: string, additive = false) => {\n    if (!additive && entityId === selectedEntityIdRef.current && selectedEntityIds.size === 1) return;",
    "  const selectEntity = async (entityId: string, additive = false) => {\n"
    "    setInspectorTarget('entity');\n"
    "    if (!additive && entityId === selectedEntityIdRef.current && selectedEntityIds.size === 1) return;",
)
replace(
    "  const mutateHierarchyEntity = async (type: string, payload: Record<string, unknown>) => {",
    "  const selectWorld = () => {\n"
    "    setInspectorTarget('world');\n"
    "    if (activeTool === 'terrain') setActiveTool('select');\n"
    "  };\n\n"
    "  const mutateHierarchyEntity = async (type: string, payload: Record<string, unknown>) => {",
)
replace(
    "    const parent = selectedSnapshot?.entity;\n    void mutateHierarchyEntity('entity.create', { kind, ...(parent ? { parent } : {}) });",
    "    const parent = inspectorTarget === 'entity' ? selectedSnapshot?.entity : undefined;\n"
    "    void mutateHierarchyEntity('entity.create', { kind, ...(parent ? { parent } : {}) });",
)
replace(
    "    if (!selectedSnapshot || !window.arc?.dialog?.createPrefab) {",
    "    if (inspectorTarget !== 'entity' || !selectedSnapshot || !window.arc?.dialog?.createPrefab) {",
)
replace(
    "          project={project}\n          selectedEntityId={selectedEntityId}\n          selectedEntityIds={selectedEntityIds}\n          onSelectEntity={selectEntity}",
    "          project={project}\n"
    "          worldSelected={inspectorTarget === 'world'}\n"
    "          selectedEntityId={inspectorTarget === 'entity' ? selectedEntityId : ''}\n"
    "          selectedEntityIds={inspectorTarget === 'entity' ? selectedEntityIds : new Set()}\n"
    "          onSelectWorld={selectWorld}\n"
    "          onSelectEntity={selectEntity}",
)
replace(
    "    if (activeTool !== 'terrain' || !selectedSnapshot?.terrain || !project) return undefined;",
    "    if (inspectorTarget !== 'entity' || activeTool !== 'terrain' || !selectedSnapshot?.terrain || !project)\n"
    "      return undefined;",
)
replace(
    "          terrainEnabled={selectedSnapshot?.terrain !== null && selectedSnapshot?.terrain !== undefined}",
    "          terrainEnabled={\n"
    "            inspectorTarget === 'entity' && selectedSnapshot?.terrain !== null && selectedSnapshot?.terrain !== undefined\n"
    "          }",
)
replace(
    """  const renderRightPanel = (panel: WorkbenchPanelId) => {
    if (panel === 'inspector') {
      if (activeTool === 'terrain' && selectedSnapshot?.terrain) {
""",
    """  const renderRightPanel = (panel: WorkbenchPanelId) => {
    if (panel === 'inspector') {
      if (inspectorTarget === 'world') {
        return (
          <WorldInspectorPanel
            environment={worldEnvironment}
            onEnvironmentChange={updateWorldEnvironment}
            assets={project?.assets ?? []}
            thumbnailProvider={loadAssetThumbnail}
            onEnvironmentPreset={applyWorldEnvironmentPreset}
            onEnvironmentHdri={applyWorldEnvironmentHdri}
          />
        );
      }
      if (activeTool === 'terrain' && selectedSnapshot?.terrain) {
""",
)
replace(
    """    if (panel === 'worldSettings') {
      return (
        <WorldSettingsPanel
          environment={worldEnvironment}
          onEnvironmentChange={updateWorldEnvironment}
          assets={project?.assets ?? []}
          thumbnailProvider={loadAssetThumbnail}
          onEnvironmentPreset={applyWorldEnvironmentPreset}
          onEnvironmentHdri={applyWorldEnvironmentHdri}
        />
      );
    }
    return <LightingPanel entities={project?.scene ?? []} onSelect={(id) => void selectEntity(id)} />;
  };
""",
    "    return <div className=\"tool-empty\">Panel unavailable.</div>;\n  };\n",
)
replace(
    "          parent={selectedSnapshot?.entity}",
    "          parent={inspectorTarget === 'entity' ? selectedSnapshot?.entity : undefined}",
)
replace(
    "export function ExplorerPanel({\n  project,\n  selectedEntityId,",
    "export function ExplorerPanel({\n  project,\n  worldSelected,\n  selectedEntityId,",
)
replace(
    "  selectedEntityIds,\n  onSelectEntity,",
    "  selectedEntityIds,\n  onSelectWorld,\n  onSelectEntity,",
)
replace(
    "  project: ProjectSnapshot;\n  selectedEntityId: string;",
    "  project: ProjectSnapshot;\n  worldSelected: boolean;\n  selectedEntityId: string;",
)
replace(
    "  selectedEntityIds: ReadonlySet<string>;\n  onSelectEntity:",
    "  selectedEntityIds: ReadonlySet<string>;\n  onSelectWorld: () => void;\n  onSelectEntity:",
)
replace(
    "  const selectedCount = selectedEntityIds.size;",
    "  const selectedCount = worldSelected ? 0 : selectedEntityIds.size;",
)
replace(
    "          <UiIconButton label=\"Duplicate selected entity\" onClick={onDuplicate}>",
    "          <UiIconButton disabled={worldSelected || selectedCount === 0} label=\"Duplicate selected entity\" onClick={onDuplicate}>",
)
replace(
    "          <UiIconButton label=\"Create prefab from selection\" onClick={onCreatePrefab}>",
    "          <UiIconButton disabled={worldSelected || selectedCount === 0} label=\"Create prefab from selection\" onClick={onCreatePrefab}>",
)
replace(
    "          <UiIconButton label=\"Delete selected entity\" onClick={onDelete}>",
    "          <UiIconButton disabled={worldSelected || selectedCount === 0} label=\"Delete selected entity\" onClick={onDelete}>",
)
replace(
    """        <div className="hierarchy-tree">
          {visibleScene.map((entity) => (
""",
    """        <div className="hierarchy-tree">
          <UiTreeRow
            as="div"
            role="treeitem"
            tabIndex={0}
            className="tree-row entity-row hierarchy-world-row"
            depth={0}
            selected={worldSelected}
            onClick={onSelectWorld}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                onSelectWorld();
              }
            }}
          >
            <span className="hierarchy-expand">
              <ChevronRight size={13} className="ghost" />
            </span>
            <Globe2 className="entity-icon entity-icon-world" size={14} />
            <span>World</span>
          </UiTreeRow>
          {visibleScene.map((entity) => (
""",
)
replace("function WorldSettingsPanel({", "function WorldInspectorPanel({")
replace(
    "          icon={<Settings />}\n          title=\"World Settings\"",
    "          icon={<Globe2 />}\n          title=\"World\"",
)

path.write_text(text)
print('Applied Workbench world inspector refactor')
