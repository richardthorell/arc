const viewportMenuButtonSelector = '.arc-viewport-show-popup button';
const viewportMenuSummarySelector = 'details.arc-viewport-show-menu > summary';
const viewportStatsToggleSelector =
  '.arc-viewport-view-options.compact > button[title^="Frame selected"], .arc-viewport-view-options.compact > button[data-viewport-stats-toggle="true"]';

const mebibyte = 1024 * 1024;
const viewportTelemetryVersion = 1;

type RuntimeViewportStats = {
  viewportId?: string;
  viewportTelemetryVersion?: number;
  fps?: number;
  frameTimeMs?: number;
  frameIntervalMs?: number;
  cpuRenderTimeMs?: number;
  drawCalls?: number;
  triangles?: number;
  triangleCount?: number;
  vertices?: number;
  vertexCount?: number;
  verticesComplete?: boolean;
  gpuMemoryMb?: number;
  memoryMb?: number;
  gpuMemoryBytes?: number;
  memoryBytes?: number;
  gpuMemoryBudgetBytes?: number;
};

const statsPollers = new WeakMap<Element, number>();
const staleTelemetryWarnings = new WeakSet<Element>();

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

const compactMemoryBytes = (bytes: number | undefined) => {
  if (typeof bytes !== 'number' || !Number.isFinite(bytes)) return '—';
  const memory = Math.max(0, bytes);
  if (memory >= 1024 ** 3)
    return `${(memory / 1024 ** 3).toFixed(2).replace(/0$/, '').replace(/\.0$/, '')} GB`;
  return `${Math.round(memory / mebibyte)} MB`;
};

const compactMemoryUsage = (usedBytes: number | undefined, budgetBytes: number | undefined) => {
  const used = compactMemoryBytes(usedBytes);
  if (used === '—') return 'N/A';
  const budget = compactMemoryBytes(budgetBytes);
  return budget === '—' ? used : `${used} / ${budget}`;
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
    setStatsCardValue(card, 'Memory', compactMemoryBytes(4280 * mebibyte));
  }
};

const viewportIdForShell = (shell: Element) => {
  const element = shell as HTMLElement;
  if (element.dataset.viewportId) return element.dataset.viewportId;

  const title = shell.querySelector('.arc-viewport-title span')?.textContent ?? 'Viewport 1';
  const index = Number.parseInt(title.match(/\d+/)?.[0] ?? '1', 10);
  const viewportId = `viewport-${Number.isFinite(index) ? index : 1}`;
  element.dataset.viewportId = viewportId;
  return viewportId;
};

const showTelemetryVersionMismatch = (shell: Element, card: Element) => {
  setStatsCardValue(card, 'Triangles', 'Restart host');
  setStatsCardValue(card, 'Vertices', 'Restart host');
  setStatsCardValue(card, 'Memory', 'Restart host');
  if (staleTelemetryWarnings.has(shell)) return;
  staleTelemetryWarnings.add(shell);
  console.warn(
    `Viewport telemetry requires arc_host_process schema ${viewportTelemetryVersion}. Rebuild and restart the native editor host.`,
  );
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
    if (typeof payload.viewportId === 'string' && payload.viewportId) {
      (shell as HTMLElement).dataset.viewportId = payload.viewportId;
    }
    if (payload.viewportTelemetryVersion !== viewportTelemetryVersion) {
      showTelemetryVersionMismatch(shell, card);
      return;
    }

    staleTelemetryWarnings.delete(shell);
    const triangles = payload.triangles ?? payload.triangleCount;
    const vertices = payload.vertices ?? payload.vertexCount;
    const memoryBytes =
      payload.gpuMemoryBytes ??
      payload.memoryBytes ??
      (typeof payload.gpuMemoryMb === 'number' ? payload.gpuMemoryMb * mebibyte : undefined) ??
      (typeof payload.memoryMb === 'number' ? payload.memoryMb * mebibyte : undefined);
    const frameIntervalMs =
      payload.frameIntervalMs ??
      (typeof payload.fps === 'number' && payload.fps > 0 ? 1000 / payload.fps : payload.frameTimeMs);

    if (typeof payload.fps === 'number' && Number.isFinite(payload.fps))
      setStatsCardValue(card, 'FPS', payload.fps > 0 ? payload.fps.toFixed(0) : '—');
    if (typeof frameIntervalMs === 'number' && Number.isFinite(frameIntervalMs))
      setStatsCardValue(card, 'Frame Time', frameIntervalMs > 0 ? `${frameIntervalMs.toFixed(2)} ms` : '—');
    if (typeof payload.drawCalls === 'number') setStatsCardValue(card, 'Draw Calls', compactCount(payload.drawCalls));
    setStatsCardValue(card, 'Triangles', compactCount(triangles));
    setStatsCardValue(card, 'Vertices', payload.verticesComplete === false ? 'N/A' : compactCount(vertices));
    setStatsCardValue(card, 'Memory', compactMemoryUsage(memoryBytes, payload.gpuMemoryBudgetBytes));
  } catch {
    // The existing ViewportPanel handles host errors. Statistics are optional UI.
  }
};

const stopRuntimeStatsPolling = (shell: Element) => {
  const poller = statsPollers.get(shell);
  if (poller !== undefined) window.clearInterval(poller);
  statsPollers.delete(shell);
};

const startRuntimeStatsPolling = (shell: Element) => {
  stopRuntimeStatsPolling(shell);
  void refreshRuntimeStats(shell);
  const poller = window.setInterval(() => {
    if (!shell.isConnected || !shell.classList.contains('show-stats')) {
      stopRuntimeStatsPolling(shell);
      return;
    }
    void refreshRuntimeStats(shell);
  }, 500);
  statsPollers.set(shell, poller);
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
      if (visible) startRuntimeStatsPolling(shell);
      else stopRuntimeStatsPolling(shell);
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
