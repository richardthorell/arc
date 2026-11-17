/** An authoritative CMake File API codemodel dependency slice for the requested build target. */
export type BuildTargetNode = {
  id: string;
  name: string;
  type: string;
  /** Native CMake target IDs, never inferred from console output. */
  dependencies: string[];
  /** Native build artifact paths returned by CMake (relative to the build directory or absolute). */
  artifacts: string[];
  /** Only set when a matching artifact is explicitly reported by the build system. */
  artifactObservedAt?: string;
  /** Set for the requested target when cmake --build returns success; it may be up to date. */
  satisfiedAt?: string;
};

export type BuildTargetGraph = {
  revision: number;
  generator: string;
  configuration: string;
  rootTargetId: string;
  discoveredAt: string;
  nodes: BuildTargetNode[];
  /** If the reachable graph exceeded the bounded UI slice. */
  truncated: boolean;
};
