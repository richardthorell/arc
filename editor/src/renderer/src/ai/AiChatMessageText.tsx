import type { MouseEvent, ReactNode } from 'react';

const bareUrlPattern = /https?:\/\/[^\s<>"']+/giu;
const trailingUrlPunctuation = /[.,!?;:)}\]]+$/u;

const splitTrailingPunctuation = (value: string): { url: string; trailing: string } => {
  const trailing = value.match(trailingUrlPunctuation)?.[0] ?? '';
  return {
    url: trailing ? value.slice(0, -trailing.length) : value,
    trailing,
  };
};

const safeHttpUrl = (value: string): string | null => {
  try {
    const parsed = new URL(value);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? parsed.toString() : null;
  } catch {
    return null;
  }
};

const linkClick = (event: MouseEvent<HTMLAnchorElement>) => {
  if (!event.ctrlKey && !event.metaKey) event.preventDefault();
};

export const renderAiChatMessageText = (text: string): ReactNode => {
  const nodes: ReactNode[] = [];
  let offset = 0;

  for (const match of text.matchAll(bareUrlPattern)) {
    const start = match.index ?? 0;
    const raw = match[0];
    if (start > offset) nodes.push(text.slice(offset, start));

    const { url, trailing } = splitTrailingPunctuation(raw);
    const href = safeHttpUrl(url);
    if (href) {
      nodes.push(
        <a
          className="ai-chat-message-link"
          href={href}
          key={`${start}-${href}`}
          onClick={linkClick}
          rel="noreferrer noopener"
          target="_blank"
          title="Ctrl+click to open link"
        >
          {url}
        </a>,
      );
      if (trailing) nodes.push(trailing);
    } else {
      nodes.push(raw);
    }
    offset = start + raw.length;
  }

  if (offset < text.length) nodes.push(text.slice(offset));
  return nodes.length ? nodes : text;
};
