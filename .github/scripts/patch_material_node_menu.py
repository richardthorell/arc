from pathlib import Path

path = Path('editor/src/renderer/src/material/MaterialGraphEditor.tsx')
text = path.read_text()
text = text.replace(
    "import { ChevronLeft, ChevronRight, Copy, Plus, Search, Trash2 } from 'lucide-react';",
    "import { ChevronRight, Copy, Plus, Search, Trash2 } from 'lucide-react';",
)

old = '''            ) : !nodeMenuCategory ? (
              visibleCategories.map((category) => (
                <UiContextMenuItem
                  key={category}
                  onClick={() => {
                    setNodeMenuCategory(category);
                    setNodeMenuSubcategory(null);
                  }}
                  onMouseEnter={() => {
                    setNodeMenuCategory(category);
                    setNodeMenuSubcategory(null);
                  }}
                  trailing={<ChevronRight size={13} />}
                >
                  <strong>{category}</strong>
                </UiContextMenuItem>
              ))
            ) : !nodeMenuSubcategory ? (
              <>
                <UiContextMenuItem onClick={() => setNodeMenuCategory(null)} leading={<ChevronLeft size={13} />}>
                  All Categories
                </UiContextMenuItem>
                {visibleSubcategories.map((subcategory) => (
                  <UiContextMenuItem
                    key={subcategory}
                    onClick={() => setNodeMenuSubcategory(subcategory)}
                    onMouseEnter={() => setNodeMenuSubcategory(subcategory)}
                    trailing={<ChevronRight size={13} />}
                  >
                    <strong>{subcategory}</strong>
                  </UiContextMenuItem>
                ))}
              </>
            ) : (
              <>
                <UiContextMenuItem onClick={() => setNodeMenuSubcategory(null)} leading={<ChevronLeft size={13} />}>
                  {nodeMenuCategory}
                </UiContextMenuItem>
                {visibleCategoryNodes.map((definition) => (
                  <UiContextMenuItem key={definition.type} onClick={() => addNode(definition.type)}>
                    <strong>{definition.title}</strong>
                  </UiContextMenuItem>
                ))}
              </>
            )}'''

new = '''            ) : (
              visibleCategories.map((category) => {
                const categoryActive = nodeMenuCategory === category;
                const categorySubcategories = materialNodeSubcategoryOrder[category].filter((subcategory) =>
                  availableNodes.some(
                    (definition) => definition.category === category && definition.subcategory === subcategory,
                  ),
                );
                const submenuDirection =
                  addMenu && canvasRef.current && addMenu.screen[0] + 280 + 2 * 240 > canvasRef.current.clientWidth
                    ? 'left'
                    : 'right';

                return (
                  <div
                    className={`material-node-menu-cascade-entry material-node-menu-cascade-${submenuDirection}`}
                    key={category}
                    onMouseEnter={() => {
                      setNodeMenuCategory(category);
                      setNodeMenuSubcategory(null);
                    }}
                  >
                    <UiContextMenuItem
                      aria-expanded={categoryActive}
                      aria-haspopup="menu"
                      onClick={() => {
                        setNodeMenuCategory(category);
                        setNodeMenuSubcategory(null);
                      }}
                      trailing={<ChevronRight size={13} />}
                    >
                      <strong>{category}</strong>
                    </UiContextMenuItem>
                    {categoryActive && (
                      <UiContextMenu
                        aria-label={`${category} material node categories`}
                        className="material-node-menu-submenu"
                        maxHeight={380}
                        width={240}
                      >
                        {categorySubcategories.map((subcategory) => {
                          const subcategoryActive = nodeMenuSubcategory === subcategory;
                          const subcategoryNodes = availableNodes.filter(
                            (definition) =>
                              definition.category === category && definition.subcategory === subcategory,
                          );
                          return (
                            <div
                              className={`material-node-menu-cascade-entry material-node-menu-cascade-${submenuDirection}`}
                              key={subcategory}
                              onMouseEnter={() => setNodeMenuSubcategory(subcategory)}
                            >
                              <UiContextMenuItem
                                aria-expanded={subcategoryActive}
                                aria-haspopup="menu"
                                onClick={() => setNodeMenuSubcategory(subcategory)}
                                trailing={<ChevronRight size={13} />}
                              >
                                <strong>{subcategory}</strong>
                              </UiContextMenuItem>
                              {subcategoryActive && (
                                <UiContextMenu
                                  aria-label={`${subcategory} material nodes`}
                                  className="material-node-menu-submenu"
                                  maxHeight={380}
                                  width={240}
                                >
                                  {subcategoryNodes.map((definition) => (
                                    <UiContextMenuItem key={definition.type} onClick={() => addNode(definition.type)}>
                                      <strong>{definition.title}</strong>
                                    </UiContextMenuItem>
                                  ))}
                                </UiContextMenu>
                              )}
                            </div>
                          );
                        })}
                      </UiContextMenu>
                    )}
                  </div>
                );
              })
            )}'''

if old not in text:
    raise SystemExit('material menu block not found')
text = text.replace(old, new)
path.write_text(text)

css_path = Path('editor/src/renderer/src/material/materialEditor.css')
css = css_path.read_text()
old_css = '''.material-node-menu.ui-context-menu {
  z-index: 20;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  padding: 0;
}'''
new_css = '''.material-node-menu.ui-context-menu {
  z-index: 20;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  padding: 0;
  overflow: visible;
}

.material-node-menu-cascade-entry {
  position: relative;
}

.material-node-menu-submenu.ui-context-menu {
  z-index: 22;
  top: -4px;
  display: block;
  padding: 4px;
  overflow-x: visible;
  overflow-y: auto;
}

.material-node-menu-cascade-right > .material-node-menu-submenu {
  left: calc(100% - 2px);
}

.material-node-menu-cascade-left > .material-node-menu-submenu {
  right: calc(100% - 2px);
}'''
if old_css not in css:
    raise SystemExit('material menu css block not found')
css = css.replace(old_css, new_css)
css = css.replace(
    '''.material-node-menu-items {
  min-height: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
  padding: 4px;
}''',
    '''.material-node-menu-items {
  min-height: 0;
  overflow: visible;
  overscroll-behavior: contain;
  padding: 4px;
}''',
)
css_path.write_text(css)

test_path = Path('editor/src/renderer/src/material/MaterialGraphEditor.test.tsx')
tests = test_path.read_text()
tests = tests.replace(
    "    const item = within(menu).getByRole('menuitem', { name: /Constant/ });",
    "    const constantsMenu = screen.getByRole('menu', { name: 'Constants material nodes' });\n    const item = within(constantsMenu).getByRole('menuitem', { name: 'Constant' });",
)
old_test = '''  it('opens material categories and subcategories on hover', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Add Node' }));
    const menu = screen.getByRole('menu', { name: 'Add material node' });
    const math = within(menu).getByRole('menuitem', { name: /Math/ });
    expect(within(menu).queryByRole('menuitem', { name: /Arithmetic/ })).not.toBeInTheDocument();

    fireEvent.mouseEnter(math);
    expect(within(menu).getByRole('menuitem', { name: /Arithmetic/ })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: /Trigonometry/ })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: /Measurement/ })).toBeInTheDocument();

    fireEvent.mouseEnter(within(menu).getByRole('menuitem', { name: /Arithmetic/ }));
    expect(within(menu).getByRole('menuitem', { name: 'Add' })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: /Fmod/ })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: /One Minus/ })).toBeInTheDocument();
  });'''
new_test = '''  it('opens material categories and subcategories as cascading side menus', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Add Node' }));
    const menu = screen.getByRole('menu', { name: 'Add material node' });
    const math = within(menu).getByRole('menuitem', { name: /Math/ });
    expect(screen.queryByRole('menu', { name: 'Math material node categories' })).not.toBeInTheDocument();

    fireEvent.mouseEnter(math.closest('.material-node-menu-cascade-entry')!);
    const categoryMenu = screen.getByRole('menu', { name: 'Math material node categories' });
    expect(categoryMenu).toHaveClass('material-node-menu-submenu');
    expect(within(menu).getByRole('menuitem', { name: /Values/ })).toBeInTheDocument();
    expect(within(categoryMenu).getByRole('menuitem', { name: /Arithmetic/ })).toBeInTheDocument();
    expect(within(categoryMenu).getByRole('menuitem', { name: /Trigonometry/ })).toBeInTheDocument();
    expect(within(categoryMenu).getByRole('menuitem', { name: /Measurement/ })).toBeInTheDocument();

    const arithmetic = within(categoryMenu).getByRole('menuitem', { name: /Arithmetic/ });
    fireEvent.mouseEnter(arithmetic.closest('.material-node-menu-cascade-entry')!);
    const commandMenu = screen.getByRole('menu', { name: 'Arithmetic material nodes' });
    expect(commandMenu).toHaveClass('material-node-menu-submenu');
    expect(within(commandMenu).getByRole('menuitem', { name: 'Add' })).toBeInTheDocument();
    expect(within(commandMenu).getByRole('menuitem', { name: /Fmod/ })).toBeInTheDocument();
    expect(within(commandMenu).getByRole('menuitem', { name: /One Minus/ })).toBeInTheDocument();
  });'''
if old_test not in tests:
    raise SystemExit('material menu test block not found')
tests = tests.replace(old_test, new_test)
test_path.write_text(tests)
