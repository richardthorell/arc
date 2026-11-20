import type { Writable } from 'node:stream';
import { livePlayMessageSchema, type LivePlayAuthority, type LivePlayCapability } from '../common/livePlayProtocol';
import { LivePlayControlServer } from './livePlayControlServer';

export type NativeLivePlayLaunchContext = {
  authority: LivePlayAuthority;
  schemaHash: string;
  capabilities: readonly LivePlayCapability[];
};

/**
 * Encode the one-shot protected inherited-pipe handoff for the native
 * Development runtime. Only trusted main-process launch code may call this.
 * The secret is NEVER exposed to renderer IPC, argv, URLs, files or logs.
 */
export function encodeLivePlayNativeHandoff(
  server: LivePlayControlServer,
  context: NativeLivePlayLaunchContext,
): Buffer {
  const handoff = server.launchHandoff();
  try {
    if (
      !livePlayMessageSchema.safeParse({
        protocol: 'arc.live-play',
        protocolVersion: 1,
        sender: 'runtime',
        ...context.authority,
        sequence: 1,
        messageId: 'native-launch-preflight',
        type: 'hello',
        payload: {
          schemaHash: context.schemaHash,
          lastReceivedSequence: 0,
          capabilities: context.capabilities,
        },
      }).success
    ) {
      throw new Error('Invalid Live Play native launch context');
    }
    const data = Buffer.from(
      JSON.stringify({
        port: handoff.port,
        secret: handoff.secret.toString('hex'),
        projectGuid: context.authority.projectGuid,
        buildId: context.authority.buildId,
        targetProfile: context.authority.targetProfile,
        sessionId: context.authority.sessionId,
        instanceId: context.authority.instanceId,
        schemaHash: context.schemaHash,
        capabilities: context.capabilities,
      }),
      'utf8',
    );
    if (data.length === 0 || data.length > 4096) {
      data.fill(0);
      throw new Error('Invalid Live Play native handoff size');
    }
    const frame = Buffer.allocUnsafe(4 + data.length);
    frame.writeUInt32BE(data.length, 0);
    data.copy(frame, 4);
    data.fill(0);
    return frame;
  } finally {
    handoff.secret.fill(0);
  }
}

/**
 * The #1140 launch adapter supplies a privileged write-end of an anonymous
 * inherited pipe (not stdout, stderr or a named discovery file). It owns the
 * child process and closes its pipe after handoff. The buffer is erased once
 * the native OS stream accepts it.
 */
export async function writeLivePlayNativeHandoff(
  writer: Writable,
  server: LivePlayControlServer,
  context: NativeLivePlayLaunchContext,
): Promise<void> {
  const frame = encodeLivePlayNativeHandoff(server, context);
  try {
    await new Promise<void>((resolve, reject) => {
      writer.write(frame, (error?: Error | null) => (error ? reject(error) : resolve()));
    });
  } finally {
    frame.fill(0);
  }
}
