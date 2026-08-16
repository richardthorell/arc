const viewportMenuButtonSelector = '.arc-viewport-show-popup button';
const viewportMenuSummarySelector = 'details.arc-viewport-show-menu > summary';
const viewportStatsToggleSelector =
  '.arc-viewport-view-options.compact > button[title^="Frame selected"], .arc-viewport-view-options.compact > button[data-viewport-stats-toggle="true"]';

type RuntimeViewportStats = {
  triangles?: number;
  triangleCount?: number;
  vertices?: number;
  vertexCount?: number;
  gpuMemoryMb?: number;
  memoryMb?: number;
  gpuMemoryBytes?: number;
  memoryBytes?: number;
};

const decorateStatsToggle = (button: HTMLButtonElement) => {
  button.dataset.viewportStatsToggle = 'true';
  button.title = 'Toggle viewport statistics';
  button.setAttribute('aria-label', 'Toggle viewport statistics');
  button.setAttribute('aria-pressed', button.closest('.arc-viewport-shell')?.classList.contains('show-stats') ? 'true' : 'false');
};

const compactCount = (value: number | undefined) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  const count = Math.max(0, value);
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(2).replace(/\.00$/, '')}M`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1).replace(/\.0$/, '')}K`;
  return count.toLocaleString();
};

const compactMemory = (megabytes: number | undefined) => {
  if (typeof megabytes !== 'number' || !Number.isFinite(megabytes)) return '—';
  const memory = Math.max(0, megabytes);
  if (memory >= 1024) return `${(memory / 1024).toFixed(2).replace(/0$/, '').replace(/\.0$/, '')} GB`;
  return `${Math.round(memory)} MB`;
};

const ensureStatsCard = (shell: Element) => {
  const controls = shell.querySelector<HTMLElement>('.arc-viewport-view-options.compact');
  if (!controls) return null;

  let card = controls.querySelector<HTMLElement>('.arc-viewport-stats-card');
  if (!card) {
    card = document.createElement('div');
    card.className = 'arc-viewport-stats-card';
    card.setAttribute('aria-label', 'Viewport statistics');
    card.innerHTML = ['FPS', 'Frame Time', 'Draw Calls', 'Triangles', 'Vertices', 'Memory']
      .map(
        (label) =>
          `<div class="arc-viewport-stats-row" data-stat-row="${label}"><span>${label}</span><strong>—</strong></div>`,
      )
      .join('');
    controls.append(card);
  }
  return card;
};

const setStatsCardValue = (card: Element, label: string, value: string) => {
  const output = card.querySelector<HTMLElement>(`.arc-viewport-stats-row[data-stat-row="${label}"] strong`);
  if (output && output.textContent !== value) output.textContent = value;
};

const decorateStatsRows = (shell: Element) => {
  const card = ensureStatsCard(shell);
  if (!card) return;

  const rows = Array.from(shell.querySelectorAll<HTMLElement>('.arc-viewport-header-stat'));
  const fps = rows[0]?.textContent?.trim().replace(/\s*FPS$/i, '') || '—';
  const frameTime = rows[1]?.textContent?.trim() || '—';
  const drawCalls = rows[2]?.textContent?.trim().replace(/\s*draws?$/i, '') || '—';
  setStatsCardValue(card, 'FPS', fps);
  setStatsCardValue(card, 'Frame Time', frameTime);
  setStatsCardValue(card, 'Draw Calls', drawCalls);

  // UI Lab has no renderer host. Use deterministic fixture values so all six
  // rows can be judged visually; production values come from viewport.state.
  if (shell.closest('.ui-lab-production-panel')) {
    setStatsCardValue(card, 'Triangles', compactCount(3_840_220));
    setStatsCardValue(card, 'Vertices', compactCount(6_120_480));
    setStatsCardValue(card, 'Memory', compactMemory(4280));
  }
};

const viewportIdForShell = (shell: Element) => {
  const title = shell.querySelector('.arc-viewport-title span')?.textContent ?? 'Viewport 1';
  const index = Number.parseInt(title.match(/\d+/)?.[0] ?? '1', 10);
  return `viewport-${Number.isFinite(index) ? index : 1}`;
};

const refreshRuntimeStats = async (shell: Element) => {
  if (!window.arc?.host?.query || shell.closest('.ui-lab-production-panel')) return;
  try {
    const response = (await window.arc.host.query('viewport.state', {
      viewportId: viewportIdForShell(shell),
    })) as { succeeded?: boolean; payload?: RuntimeViewportStats };
    if (response?.succeeded === false || !response?.payload) return;

    const card = ensureStatsCard(shell);
    if (!card) return;
    const payload = response.payload;
    const triangles = payload.triangles ?? payload.triangleCount;
    const vertices = payload.vertices ?? payload.vertexCount;
    const memoryMb =
      payload.gpuMemoryMb ??
      payload.memoryMb ??
      (typeof payload.gpuMemoryBytes === 'number' ? payload.gpuMemoryBytes / (1024 * 1024) : undefined) ??
      (typeof payload.memoryBytes === 'number' ? payload.memoryBytes / (1024 * 1024) : undefined);

    setStatsCardValue(card, 'Triangles', compactCount(triangles));
    setStatsCardValue(card, 'Vertices', compactCount(vertices));
    setStatsCardValue(card, 'Memory', compactMemory(memoryMb));
  } catch {
    // The existing ViewportPanel handles host errors. Statistics are optional UI.
  }
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

    const summary = target.closest<HTMLElement>(viewportMenuSummarySelector);
    if (summary) {
      const selectedMenu = summary.closest<HTMLDetailsElement>('details.arc-viewport-show-menu');
      const shell = selectedMenu?.closest('.arc-viewport-shell');
      shell?.querySelectorAll<HTMLDetailsElement>('details.arc-viewport-show-menu[open]').forEach((menu) => {
        if (menu !== selectedMenu) menu.open = false;
      });
      return;
    }

    const statsToggle = target.closest<HTMLButtonElement>(viewportStatsToggleSelector);
    if (statsToggle) {
      event.preventDefault();
      event.stopPropagation();
      const shell = statsToggle.closest('.arc-viewport-shell');
      if (!shell) return;
      decorateStatsRows(shell);
      const visible = shell.classList.toggle('show-stats');
      statsToggle.setAttribute('aria-pressed', visible ? 'true' : 'false');
      if (visible) void refreshRuntimeStats(shell);
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
