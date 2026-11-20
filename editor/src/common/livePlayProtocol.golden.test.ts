import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { livePlayMessageSchema, receiveLivePlayFrame } from './livePlayProtocol';

const fixtures = JSON.parse(
  readFileSync(new URL('../../../engine/network/test_support/live_play_wire_v1.json', import.meta.url), 'utf8'),
) as Record<string, unknown>;

describe('Live Play L1.2b cross-language golden contract', () => {
  it('accepts the exact native C++ hello envelope as an authenticated runtime peer', () => {
    const message = livePlayMessageSchema.parse(fixtures.hello);
    expect(message.type).toBe('hello');
    const received = receiveLivePlayFrame(JSON.stringify(fixtures.hello), {
      authority: {
        projectGuid: '11111111-1111-1111-1111-111111111111',
        buildId: 'dev-build-42',
        targetProfile: 'windows-x64-vulkan',
        sessionId: 'session-1',
        instanceId: 'runtime-1',
      },
      expectedSchemaHash: 'a'.repeat(64),
      authenticated: true,
      peerRole: 'runtime',
      lastSequence: 0,
      grantedCapabilities: ['session.observe', 'content.observe'],
    });
    expect(received).toMatchObject({ accepted: true, nextSequence: 1 });
  });

  it('validates the exact heartbeat frame consumed by the native client', () => {
    const value = livePlayMessageSchema.parse(fixtures.editorHeartbeat);
    expect(value).toMatchObject({
      sender: 'editor',
      type: 'heartbeat',
      sequence: 2,
      payload: { lastReceivedSequence: 2 },
    });
  });
});
