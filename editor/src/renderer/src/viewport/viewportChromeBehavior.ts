const viewportMenuButtonSelector = '.arc-viewport-show-popup button';
const viewportStatsToggleSelector =
  '.arc-viewport-view-options.compact > button[title^="Frame selected"], .arc-viewport-view-options.compact > button[data-viewport-stats-toggle="true"]';

const decorateStatsToggle = (button: HTMLButtonElement) => {
  button.dataset.viewportStatsToggle = 'true';
  button.title = 'Toggle viewport statistics';
  button.setAttribute('aria-label', 'Toggle viewport statistics');
  button.setAttribute('aria-pressed', button.closest('.arc-viewport-shell')?.classList.contains('show-stats') ? 'true' : 'false');
};

const decorateStatsRows = (shell: Element) => {
  const rows = Array.from(shell.querySelectorAll<HTMLElement>('.arc-viewport-header-stat'));
  const definitions = [
    ['FPS', (value: string) => value.replace(/\s*FPS$/i, '')],
    ['Frame Time', (value: string) => value],
    ['Draw Calls', (value: string) => value.replace(/\s*draws?$/i, '')],
  ] as const;

  definitions.forEach(([label, format], index) => {
    const row = rows[index];
    if (!row) return;
    const rawValue = row.textContent?.trim() ?? '--';
    const value = format(rawValue);
    if (row.dataset.statLabel !== label) row.dataset.statLabel = label;
    if (row.dataset.statValue !== value) row.dataset.statValue = value;
  });
};

const decorateViewportChrome = (root: ParentNode = document) => {
  root.querySelectorAll<HTMLButtonElement>(viewportStatsToggleSelector).forEach(decorateStatsToggle);
  root.querySelectorAll('.arc-viewport-shell').forEach(decorateStatsRows);
};

decorateViewportChrome();

const observer = new MutationObserver(() => decorateViewportChrome());
observer.observe(document.documentElement, { childList: true, characterData: true, subtree: true });

document.addEventListener(
  'click',
  (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;

    const statsToggle = target.closest<HTMLButtonElement>(viewportStatsToggleSelector);
    if (statsToggle) {
      event.preventDefault();
      event.stopPropagation();
      const shell = statsToggle.closest('.arc-viewport-shell');
      if (!shell) return;
      decorateStatsRows(shell);
      const visible = shell.classList.toggle('show-stats');
      statsToggle.setAttribute('aria-pressed', visible ? 'true' : 'false');
      return;
    }

    const menuButton = target.closest<HTMLButtonElement>(viewportMenuButtonSelector);
    if (!menuButton || menuButton.disabled) return;
    const menu = menuButton.closest<HTMLDetailsElement>('details.arc-viewport-show-menu');
    if (!menu) return;

    queueMicrotask(() => {
      menu.open = false;
    });
  },
  true,
);
