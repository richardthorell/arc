// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import type { AiRuntimeRequest } from '../../../common/aiRuntimeTypes';
import { AiChatPanel } from './AiChatPanel';
import type { AiModelProvider } from './aiChat';
import { loadAiConversationStore } from './aiConversationStore';

const projectA = '11111111-1111-1111-1111-111111111111';
const projectB = '22222222-2222-2222-2222-222222222222';

const provider: AiModelProvider = {
  id: 'openai:test',
  providerId: 'openai',
  modelId: 'test',
  label: 'Test Agent',
  configured: true,
  capabilities: { streaming: true, tools: false, inputModalities: ['text'] },
  async *stream() {
    yield { type: 'delta', text: 'Saved response.' };
    yield { type: 'done', finishReason: 'stop' };
  },
};

afterEach(() => {
  cleanup();
  localStorage.clear();
});

describe('AiChatPanel project persistence', () => {
  it('restores a project conversation without exposing it to another project', async () => {
    const first = render(<AiChatPanel projectGuid={projectA} provider={provider} />);
    fireEvent.change(screen.getByLabelText('Start a conversation'), { target: { value: 'Project A question' } });
    fireEvent.click(screen.getByLabelText('Start conversation'));
    await waitFor(() => expect(screen.getByText('Saved response.')).toBeInTheDocument());
    fireEvent.click(screen.getByLabelText('Back to conversations'));
    expect(screen.getByRole('button', { name: 'Open conversation Saved response' })).toBeInTheDocument();
    first.unmount();

    const second = render(<AiChatPanel projectGuid={projectB} provider={provider} />);
    expect(screen.queryByRole('button', { name: 'Open conversation Saved response' })).not.toBeInTheDocument();
    second.unmount();

    render(<AiChatPanel projectGuid={projectA} provider={provider} />);
    expect(screen.getByRole('button', { name: 'Open conversation Saved response' })).toBeInTheDocument();
  });

  it('restores persisted active conversation and model UI state', async () => {
    const first = render(<AiChatPanel projectGuid={projectA} provider={provider} />);
    fireEvent.change(screen.getByLabelText('Start a conversation'), { target: { value: 'Resume me' } });
    fireEvent.click(screen.getByLabelText('Start conversation'));
    await waitFor(() => expect(screen.getByText('Saved response.')).toBeInTheDocument());
    first.unmount();

    render(<AiChatPanel projectGuid={projectA} provider={provider} />);
    expect(screen.getByRole('region', { name: 'Active conversation' })).toHaveTextContent('Resume me');
    expect(screen.getByLabelText('Model')).toHaveTextContent('Test Agent');
  });

  it('persists agent tool activity and replays it into a later provider turn', async () => {
    const chatRequests: AiRuntimeRequest[] = [];
    const toolProvider: AiModelProvider = {
      id: 'openai:tools',
      providerId: 'openai',
      modelId: 'tools',
      label: 'Tool Agent',
      configured: true,
      capabilities: { streaming: true, tools: true, inputModalities: ['text'] },
      async *stream(request) {
        if (request.metadata?.purpose === 'conversation-caption') {
          yield { type: 'delta', text: 'Floor inspection' };
          yield { type: 'done', finishReason: 'stop' };
          return;
        }
        chatRequests.push(request);
        yield {
          type: 'tool-call',
          call: { id: `call-${chatRequests.length}`, name: 'scene.findEntities', arguments: { search: 'Floor' } },
          agentStep: 0,
        };
        yield {
          type: 'tool-result',
          result: {
            toolCallId: `call-${chatRequests.length}`,
            name: 'scene.findEntities',
            operation: 'scene.findEntities',
            content: '{"entities":[{"guid":"floor-guid","name":"Floor"}]}',
            truncated: false,
            originalBytes: 53,
          },
          agentStep: 0,
        };
        yield { type: 'delta', text: `Tool response ${chatRequests.length}.` };
        yield { type: 'done', finishReason: 'stop' };
      },
    };

    render(<AiChatPanel projectGuid={projectA} provider={toolProvider} />);
    fireEvent.change(screen.getByLabelText('Start a conversation'), { target: { value: 'Inspect the floor' } });
    fireEvent.click(screen.getByLabelText('Start conversation'));
    await waitFor(() => expect(screen.getByText('Tool response 1.')).toBeInTheDocument());
    await waitFor(() => {
      const tool = loadAiConversationStore(projectA).conversations[0]?.messages[1]?.toolReferences?.[0];
      expect(tool).toMatchObject({
        toolCallId: 'call-1',
        name: 'scene.findEntities',
        operation: 'scene.findEntities',
        state: 'complete',
        step: 0,
        arguments: { search: 'Floor' },
        resultContent: '{"entities":[{"guid":"floor-guid","name":"Floor"}]}',
      });
    });

    fireEvent.change(screen.getByLabelText('Chat prompt'), { target: { value: 'What did you find?' } });
    fireEvent.click(screen.getByLabelText('Send prompt'));
    await waitFor(() => expect(screen.getByText('Tool response 2.')).toBeInTheDocument());

    expect(chatRequests).toHaveLength(2);
    const replayed = chatRequests[1]!.messages;
    expect(replayed.map((message) => message.role)).toEqual(['user', 'assistant', 'tool', 'assistant', 'user']);
    expect(replayed[1]?.toolCalls?.[0]).toMatchObject({
      id: 'call-1',
      name: 'scene.findEntities',
      arguments: { search: 'Floor' },
    });
    expect(replayed[2]?.toolResult).toMatchObject({
      toolCallId: 'call-1',
      operation: 'scene.findEntities',
    });
  });

  it('renders task updates in place and restores their final state after reopening', async () => {
    const taskProvider: AiModelProvider = {
      id: 'openai:tasks',
      providerId: 'openai',
      modelId: 'tasks',
      label: 'Task Agent',
      configured: true,
      capabilities: { streaming: true, tools: true, inputModalities: ['text'] },
      async *stream(request) {
        if (request.metadata?.purpose === 'conversation-caption') {
          yield { type: 'delta', text: 'Capsule build' };
          yield { type: 'done', finishReason: 'stop' };
          return;
        }
        yield {
          type: 'task-update',
          task: {
            id: 'agent-step-0',
            title: 'Configure capsule',
            state: 'in_progress',
            agentStep: 0,
            toolCallIds: ['call-capsule'],
          },
        };
        yield {
          type: 'tool-call',
          call: { id: 'call-capsule', name: 'editor.applyBatch', arguments: { operations: [] } },
          agentStep: 0,
        };
        yield {
          type: 'tool-result',
          result: {
            toolCallId: 'call-capsule',
            name: 'editor.applyBatch',
            operation: 'editor.applyBatch',
            content: '{"operationCount":1}',
          },
          agentStep: 0,
        };
        yield {
          type: 'task-update',
          task: {
            id: 'agent-step-0',
            title: 'Configure capsule',
            state: 'completed',
            agentStep: 0,
            toolCallIds: ['call-capsule'],
          },
        };
        yield { type: 'delta', text: 'Capsule configured.' };
        yield { type: 'done', finishReason: 'stop' };
      },
    };

    const first = render(<AiChatPanel projectGuid={projectA} provider={taskProvider} />);
    fireEvent.change(screen.getByLabelText('Start a conversation'), { target: { value: 'Create a capsule' } });
    fireEvent.click(screen.getByLabelText('Start conversation'));

    await waitFor(() => expect(screen.getByText('Configure capsule')).toBeInTheDocument());
    await waitFor(() =>
      expect(loadAiConversationStore(projectA).conversations[0]?.messages[1]?.taskReferences?.[0]).toMatchObject({
        id: 'agent-step-0',
        title: 'Configure capsule',
        state: 'completed',
        step: 0,
        toolCallIds: ['call-capsule'],
      }),
    );

    first.unmount();
    render(<AiChatPanel projectGuid={projectA} provider={taskProvider} />);

    expect(screen.queryByText('Configure capsule')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Show tasks' }));
    expect(screen.getByText('Configure capsule')).toBeInTheDocument();
    expect(screen.getByText('Configure capsule').closest('[data-progress-state="complete"]')).toBeInTheDocument();
  });
});
