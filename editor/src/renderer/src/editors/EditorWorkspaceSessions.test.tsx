// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { useEffect, useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { EditorDocument } from './editorTypes';
import { EditorWorkspaceSessions } from './EditorWorkspaceSessions';

const level: EditorDocument = {
  id: 'level:world',
  kind: 'level',
  title: 'World',
  dirty: false,
  readOnly: false,
};
const material: EditorDocument = {
  id: 'material:stone',
  kind: 'material',
  title: 'Stone',
  dirty: false,
  readOnly: false,
};

afterEach(cleanup);

describe('EditorWorkspaceSessions', () => {
  it('retains inactive document state and unmounts only when a document closes', () => {
    const unmounted = vi.fn();
    const Session = ({ document }: { document: EditorDocument }) => {
      const [count, setCount] = useState(0);
      useEffect(() => () => unmounted(document.id), [document.id]);
      return <button onClick={() => setCount((value) => value + 1)}>{`${document.id}:${count}`}</button>;
    };
    const renderDocument = (document: EditorDocument) => <Session document={document} />;

    const view = render(
      <EditorWorkspaceSessions
        documents={[level, material]}
        activeDocumentId={level.id}
        projectKey="project-a"
        renderDocument={renderDocument}
      />,
    );
    fireEvent.click(screen.getByText('level:world:0'));
    view.rerender(
      <EditorWorkspaceSessions
        documents={[level, material]}
        activeDocumentId={material.id}
        projectKey="project-a"
        renderDocument={renderDocument}
      />,
    );

    expect(screen.getByText('level:world:1')).toBeInTheDocument();
    expect(screen.getByText('level:world:1').parentElement).toHaveAttribute('aria-hidden', 'true');
    expect(screen.getByText('material:stone:0').parentElement).toHaveAttribute('aria-hidden', 'false');
    expect(unmounted).not.toHaveBeenCalled();

    view.rerender(
      <EditorWorkspaceSessions
        documents={[material]}
        activeDocumentId={material.id}
        projectKey="project-a"
        renderDocument={renderDocument}
      />,
    );
    expect(unmounted).toHaveBeenCalledTimes(1);
    expect(unmounted).toHaveBeenCalledWith(level.id);
  });
});
