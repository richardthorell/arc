export type PlayFromHereSource = 'camera' | 'cursor';

export type PlayFromHereTransform = {
  position: readonly [number, number, number];
  rotation: readonly [number, number, number, number];
};

export type PlayFromHereOverride = {
  version: 1;
  source: PlayFromHereSource;
  transform: PlayFromHereTransform;
};

const finiteTuple = (values: readonly number[], expectedLength: number): boolean =>
  values.length === expectedLength && values.every(Number.isFinite);

const cloneOverride = (override: PlayFromHereOverride): PlayFromHereOverride => ({
  version: 1,
  source: override.source,
  transform: {
    position: [...override.transform.position] as [number, number, number],
    rotation: [...override.transform.rotation] as [number, number, number, number],
  },
});

/**
 * Creates the transient spawn override passed to a Play session.
 *
 * The returned value is detached from editor camera/cursor state so starting Play
 * cannot mutate authored scene data or observe later editor-navigation changes.
 */
export const createPlayFromHereOverride = (
  source: PlayFromHereSource,
  transform: PlayFromHereTransform,
): PlayFromHereOverride => {
  if (!finiteTuple(transform.position, 3)) throw new Error('Play From Here position must contain three finite values');
  if (!finiteTuple(transform.rotation, 4)) throw new Error('Play From Here rotation must contain four finite values');

  return cloneOverride({ version: 1, source, transform });
};

/**
 * One-shot handoff between the editor command and the next Play-session start.
 *
 * Consuming always clears the staged override, including when the project elects
 * not to accept Play From Here. A later ordinary Play therefore cannot inherit a
 * stale editor-camera/cursor transform.
 */
export class PlayFromHereOverrideQueue {
  private staged: PlayFromHereOverride | undefined;

  stage(override: PlayFromHereOverride): void {
    this.staged = cloneOverride(override);
  }

  consume(projectAcceptsOverride: boolean): PlayFromHereOverride | undefined {
    const staged = this.staged;
    this.staged = undefined;
    return projectAcceptsOverride && staged ? cloneOverride(staged) : undefined;
  }

  clear(): void {
    this.staged = undefined;
  }

  get hasPendingOverride(): boolean {
    return this.staged !== undefined;
  }
}
