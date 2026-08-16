const viewportMenuButtonSelector = '.arc-viewport-show-popup button';
const viewportStatsToggleSelector =
  '.arc-viewport-view-options.compact > button[title^="Frame selected"], .arc-viewport-view-options.compact > button[data-viewport-stats-toggle="true"]';

const decorateStatsToggle = (button: HTMLButtonElement) => {
  button.dataset.viewportStatsToggle = 'true';
  button.title = 'Toggle viewport statistics';
  button.setAttribute('aria-label', 'Toggle viewport statistics');
  button.setAttribute('aria-pressed', button.closest('.arc-viewport-shell')?.classList.contains('show-stats') ? 'true' : 'false');
};

const decorateViewportStatsToggles = (root: ParentNode = document) => {
  root.querySelectorAll<HTMLButtonElement>(viewportStatsToggleSelector).forEach(decorateStatsToggle);
};

decorateViewportStatsToggles();

const observer = new MutationObserver(() => decorateViewportStatsToggles());
observer.observe(document.documentElement, { childList: true, subtree: true });

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
