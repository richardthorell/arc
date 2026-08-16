import { useEffect, useRef, useState } from 'react';
import {
  BookOpen,
  Bug,
  CheckCircle2,
  ChevronRight,
  ClipboardPaste,
  Copy,
  FilePlus2,
  Focus,
  FolderOpen,
  Info,
  Keyboard,
  Pencil,
  Plus,
  Redo2,
  RefreshCw,
  Save,
  Scissors,
  Search,
  Settings,
  Trash2,
  Undo2,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

import type { CommandId, WorkbenchPanelId } from '../app/workbenchTypes';
import { panelRegistry } from '../app/panelRegistry';
import { UiButton } from '../ui';
import { WindowControls } from './WindowControls';

type MenuBarProps = {
  projectTitle: string;
  onCommand: (command: CommandId) => void;
  canUndo?: boolean;
  canRedo?: boolean;
  undoLabel?: string;
  redoLabel?: string;
  gridVisible?: boolean;
  onToggleGrid?: () => void;
  onPanel?: (panel: WorkbenchPanelId) => void;
};

const menuItems = ['File', 'Edit', 'Entity', 'View', 'Window', 'Tools', 'Help'] as const;
type MenuItem = (typeof menuItems)[number];

type MenuCommand = {
  label: string;
  icon?: LucideIcon;
  command?: CommandId;
  action?: () => void;
  shortcut?: string;
  disabled?: boolean;
  checked?: boolean;
  panel?: WorkbenchPanelId;
  children?: MenuEntry[];
};

type MenuEntry = MenuCommand | { separator: true };

const separator = (): MenuEntry => ({ separator: true });
const placeholder = (label: string, options: Omit<MenuCommand, 'label' | 'disabled'> = {}): MenuCommand => ({
  label,
  disabled: true,
  ...options,
});

const baseMenuCommands: Partial<Record<MenuItem, MenuEntry[]>> = {
  File: [
    { label: 'New Scene', command: 'file.new', shortcut: 'Ctrl+N', icon: FilePlus2 },
    { label: 'Open Scene...', command: 'file.open', shortcut: 'Ctrl+O', icon: FolderOpen },
    {
      label: 'Open Recent',
      children: [placeholder('No recent scenes')],
    },
    separator(),
    { label: 'Save', command: 'file.save', shortcut: 'Ctrl+S', icon: Save },
    { label: 'Save As...', command: 'file.saveAs', shortcut: 'Ctrl+Shift+S', icon: Save },
    { label: 'Save All', command: 'assets.saveAll', shortcut: 'Ctrl+Shift+Alt+S', icon: Save },
    separator(),
    { label: 'Import Scene Into Current...', command: 'file.importScene' },
    separator(),
    placeholder('Project Settings...', { icon: Settings }),
    { label: 'Close Project', command: 'project.close' },
    placeholder('Exit ARC', { shortcut: 'Alt+F4' }),
  ],
  Entity: [
    placeholder('Create Empty Entity', { shortcut: 'Ctrl+Shift+N', icon: Plus }),
    {
      label: 'Create',
      children: [
        placeholder('Camera'),
        {
          label: 'Light',
          children: [placeholder('Directional Light'), placeholder('Point Light'), placeholder('Spot Light')],
        },
        {
          label: 'Primitive',
          children: [
            placeholder('Cube'),
            placeholder('Sphere'),
            placeholder('Plane'),
            placeholder('Cylinder'),
            placeholder('Cone'),
            placeholder('Capsule'),
          ],
        },
      ],
    },
    placeholder('Create Child Entity', { icon: Plus }),
    separator(),
    placeholder('Add Component...', { icon: Plus }),
    separator(),
    placeholder('Enable Entity', { checked: false }),
    placeholder('Clear Parent / Unparent'),
    placeholder('Reset Transform', { icon: RefreshCw }),
  ],
  Tools: [
    { label: 'Command Palette...', command: 'view.commandPalette', shortcut: 'Ctrl+Shift+P', icon: Search },
    separator(),
    placeholder('Validate Scene', { icon: CheckCircle2 }),
    placeholder('Refresh Assets', { icon: RefreshCw }),
    placeholder('Reimport All Assets'),
    separator(),
    {
      label: 'Developer',
      children: [
        placeholder('ECS Statistics'),
        placeholder('Renderer Statistics'),
        placeholder('GPU / Resource Debugger'),
        placeholder('ImGui Demo'),
      ],
    },
  ],
  Help: [
    placeholder('Documentation', { shortcut: 'F1', icon: BookOpen }),
    placeholder('Keyboard Shortcuts', { icon: Keyboard }),
    separator(),
    placeholder('ARC on GitHub'),
    placeholder('Report an Issue...', { icon: Bug }),
    separator(),
    placeholder('About ARC', { icon: Info }),
  ],
};

function MenuEntries({ entries, onRun }: { entries: MenuEntry[]; onRun: (entry: MenuCommand) => void }) {
  return (
    <>
      {entries.map((entry, index) => {
        if ('separator' in entry) {
          return <div className="menu-separator" key={`separator-${index}`} role="separator" />;
        }

        const Icon = entry.icon;
        const leading = (
          <span className="menu-leading" aria-hidden="true">
            {entry.checked !== undefined ? (
              entry.checked ? (
                '✓'
              ) : null
            ) : Icon ? (
              <Icon size={14} strokeWidth={1.8} />
            ) : null}
          </span>
        );

        if (entry.children) {
          return (
            <div className="menu-submenu-container" key={`${entry.label}-${index}`}>
              <UiButton
                disabled={entry.disabled}
                role="menuitem"
                aria-haspopup="menu"
                variant="ghost"
              >
                {leading}
                <span className="menu-entry-label">{entry.label}</span>
                <ChevronRight className="menu-submenu-chevron" size={14} aria-hidden="true" />
              </UiButton>
              <div className="menu-dropdown menu-submenu" role="menu">
                <MenuEntries entries={entry.children} onRun={onRun} />
              </div>
            </div>
          );
        }

        return (
          <UiButton
            disabled={entry.disabled}
            key={`${entry.command ?? entry.panel ?? entry.label}-${index}`}
            role={entry.checked !== undefined ? 'menuitemcheckbox' : 'menuitem'}
            aria-checked={entry.checked}
            onClick={() => onRun(entry)}
            variant="ghost"
          >
            {leading}
            <span className="menu-entry-label">{entry.label}</span>
            {entry.shortcut && <small>{entry.shortcut}</small>}
          </UiButton>
        );
      })}
    </>
  );
}

export function MenuBar({
  projectTitle,
  onCommand,
  canUndo = false,
  canRedo = false,
  undoLabel,
  redoLabel,
  gridVisible = true,
  onToggleGrid,
  onPanel,
}: MenuBarProps) {
  const [openMenu, setOpenMenu] = useState<MenuItem | null>(null);
  const menuRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const close = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) {
        setOpenMenu(null);
      }
    };

    window.addEventListener('pointerdown', close);
    return () => window.removeEventListener('pointerdown', close);
  }, []);

  const runMenuCommand = (entry: MenuCommand) => {
    setOpenMenu(null);
    if (entry.action) entry.action();
    else if (entry.panel) onPanel?.(entry.panel);
    else if (entry.command) onCommand(entry.command);
  };

  const menuCommands: Partial<Record<MenuItem, MenuEntry[]>> = {
    ...baseMenuCommands,
    Edit: [
      {
        label: undoLabel ? `Undo ${undoLabel}` : 'Undo',
        command: 'edit.undo',
        shortcut: 'Ctrl+Z',
        disabled: !canUndo,
        icon: Undo2,
      },
      {
        label: redoLabel ? `Redo ${redoLabel}` : 'Redo',
        command: 'edit.redo',
        shortcut: 'Ctrl+Shift+Z',
        disabled: !canRedo,
        icon: Redo2,
      },
      separator(),
      placeholder('Cut', { shortcut: 'Ctrl+X', icon: Scissors }),
      placeholder('Copy', { shortcut: 'Ctrl+C', icon: Copy }),
      placeholder('Paste', { shortcut: 'Ctrl+V', icon: ClipboardPaste }),
      { label: 'Duplicate', command: 'entity.duplicate', shortcut: 'Ctrl+D', icon: Copy },
      separator(),
      placeholder('Rename', { shortcut: 'F2', icon: Pencil }),
      { label: 'Delete', command: 'entity.delete', shortcut: 'Delete', icon: Trash2 },
      placeholder('Select All', { shortcut: 'Ctrl+A' }),
      separator(),
      { label: 'Preferences...', command: 'settings.open', shortcut: 'Ctrl+,', icon: Settings },
    ],
    View: [
      { label: 'Frame Selected', command: 'viewport.frameSelected', shortcut: 'F', icon: Focus },
      placeholder('Frame All', { shortcut: 'Home' }),
      separator(),
      { label: 'Grid', action: onToggleGrid, checked: gridVisible },
      placeholder('Gizmos', { checked: false }),
      placeholder('Selection Outline', { checked: false }),
      placeholder('Bounds', { checked: false }),
      separator(),
      {
        label: 'Camera',
        children: [placeholder('Perspective'), placeholder('Orthographic')],
      },
      {
        label: 'Shading',
        children: [placeholder('Lit'), placeholder('Unlit'), placeholder('Wireframe')],
      },
      separator(),
      placeholder('Maximize Active Panel', { shortcut: 'Shift+Space' }),
    ],
    Window: [
      ...(Object.values(panelRegistry) as Array<(typeof panelRegistry)[WorkbenchPanelId]>).map((panel) => ({
        label: panel.id === 'viewport' ? 'Viewport' : panel.title,
        panel: panel.id,
      })),
      separator(),
      {
        label: 'Layout',
        children: [
          { label: 'Level Design', command: 'layout.levelDesign' as CommandId },
          { label: 'Materials', command: 'layout.materials' as CommandId },
          { label: 'Profiling', command: 'layout.profiling' as CommandId },
          separator(),
          placeholder('Save Layout As...'),
          placeholder('Delete Layout...'),
          { label: 'Reset Layout', command: 'layout.reset' as CommandId },
        ],
      },
    ],
  };

  return (
    <header className="workbench-titlebar">
      <div className="titlebar-left">
        <div className="arc-app-brand" aria-label="arc editor">
          <svg className="arc-logo-mark" viewBox="0 0 28 24" role="img" aria-hidden="true">
            <path d="M2 13.2 8.2 5.4l4.1 4.2-6 8.1Z" />
            <path d="M9.1 4.1 15.6 1l10.5 9.8-4.3 4.1Z" />
            <path d="M13 13.5h6.1l6.1 7.1h-7.2Z" />
          </svg>
          <strong>arc</strong>
        </div>

        <nav ref={menuRef} className="menu-bar" aria-label="Main menu">
          {menuItems.map((item) => {
            const commands = menuCommands[item];
            const expanded = openMenu === item;
            return (
              <div key={item} className="menu-bar-item">
                <UiButton
                  aria-expanded={expanded}
                  aria-haspopup={commands ? 'menu' : undefined}
                  onClick={() => (commands ? setOpenMenu(expanded ? null : item) : setOpenMenu(null))}
                  variant="ghost"
                >
                  {item}
                </UiButton>
                {commands && expanded && (
                  <div className="menu-dropdown" role="menu">
                    <MenuEntries entries={commands} onRun={runMenuCommand} />
                  </div>
                )}
              </div>
            );
          })}
        </nav>
      </div>

      <div className="window-title" aria-label={`Current scene ${projectTitle}`}>
        <span>{projectTitle}</span>
      </div>

      <WindowControls />
    </header>
  );
}
