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

  return {
    version: 1,
    source,
    transform: {
      position: [...transform.position] as [number, number, number],
      rotation: [...transform.rotation] as [number, number, number, number],
    },
  };
};
