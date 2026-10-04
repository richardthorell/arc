import type { MouseEvent, ReactNode } from 'react';

const bareUrlPattern = /https?:\/\/[^\s<>"']+/giu;
const trailingUrlPunctuation = /[.,!?;:)}\]]+$/u;
const fencedCodePattern = /^```([^\s`]*)\s*$/u;
const headingPattern = /^(#{1,6})\s+(.+)$/u;
const unorderedListPattern = /^\s*[-+*]\s+(.+)$/u;
const orderedListPattern = /^\s*\d+[.)]\s+(.+)$/u;
const blockquotePattern = /^>\s?(.*)$/u;
const tableDividerCellPattern = /^:?-{3,}:?$/u;

type InlineToken = {
  index: number;
  length: number;
  node: (key: string) => ReactNode;
};

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
  event.preventDefault();
  if (!event.ctrlKey && !event.metaKey) return;
  window.open(event.currentTarget.href, '_blank', 'noopener,noreferrer');
};

const linkNode = (href: string, label: ReactNode, key: string, title = 'Ctrl+click to open link') => (
  <a
    className="ai-chat-message-link"
    href={href}
    key={key}
    onClick={linkClick}
    rel="noreferrer noopener"
    target="_blank"
    title={title}
  >
    {label}
  </a>
);

const firstMatch = (pattern: RegExp, value: string) => {
  pattern.lastIndex = 0;
  return pattern.exec(value);
};

const renderInlineMarkdown = (text: string, keyPrefix: string): ReactNode[] => {
  const nodes: ReactNode[] = [];
  let remaining = text;
  let offset = 0;

  while (remaining) {
    const candidates: InlineToken[] = [];

    const inlineCode = firstMatch(/`([^`\n]+)`/u, remaining);
    if (inlineCode?.index !== undefined) {
      candidates.push({
        index: inlineCode.index,
        length: inlineCode[0].length,
        node: (key) => <code key={key}>{inlineCode[1]}</code>,
      });
    }

    const markdownLink = firstMatch(/\[([^\]\n]+)\]\(([^\s)]+)\)/u, remaining);
    if (markdownLink?.index !== undefined) {
      candidates.push({
        index: markdownLink.index,
        length: markdownLink[0].length,
        node: (key) => {
          const href = safeHttpUrl(markdownLink[2]);
          return href ? linkNode(href, renderInlineMarkdown(markdownLink[1], `${key}-label`), key) : markdownLink[0];
        },
      });
    }

    const strong = firstMatch(/\*\*([^*\n]+)\*\*/u, remaining);
    if (strong?.index !== undefined) {
      candidates.push({
        index: strong.index,
        length: strong[0].length,
        node: (key) => <strong key={key}>{renderInlineMarkdown(strong[1], `${key}-strong`)}</strong>,
      });
    }

    const strike = firstMatch(/~~([^~\n]+)~~/u, remaining);
    if (strike?.index !== undefined) {
      candidates.push({
        index: strike.index,
        length: strike[0].length,
        node: (key) => <del key={key}>{renderInlineMarkdown(strike[1], `${key}-strike`)}</del>,
      });
    }

    const emphasis = firstMatch(/(?<!\*)\*([^*\n]+)\*(?!\*)/u, remaining);
    if (emphasis?.index !== undefined) {
      candidates.push({
        index: emphasis.index,
        length: emphasis[0].length,
        node: (key) => <em key={key}>{renderInlineMarkdown(emphasis[1], `${key}-em`)}</em>,
      });
    }

    const bareUrl = firstMatch(bareUrlPattern, remaining);
    if (bareUrl?.index !== undefined) {
      const raw = bareUrl[0];
      const { url, trailing } = splitTrailingPunctuation(raw);
      const href = safeHttpUrl(url);
      if (href) {
        candidates.push({
          index: bareUrl.index,
          length: raw.length,
          node: (key) => (
            <span key={key}>
              {linkNode(href, url, `${key}-link`)}
              {trailing}
            </span>
          ),
        });
      }
    }

    if (!candidates.length) {
      nodes.push(remaining);
      break;
    }

    const candidate = candidates.reduce((best, current) => (current.index < best.index ? current : best));
    if (candidate.index > 0) nodes.push(remaining.slice(0, candidate.index));
    nodes.push(candidate.node(`${keyPrefix}-${offset + candidate.index}`));

    const consumed = candidate.index + candidate.length;
    offset += consumed;
    remaining = remaining.slice(consumed);
  }

  return nodes;
};

const renderInlineWithBreaks = (text: string, keyPrefix: string): ReactNode[] => {
  const lines = text.split('\n');
  return lines.flatMap((line, index) => [
    ...renderInlineMarkdown(line, `${keyPrefix}-${index}`),
    ...(index < lines.length - 1 ? [<br key={`${keyPrefix}-br-${index}`} />] : []),
  ]);
};

const splitTableRow = (line: string) => {
  const trimmed = line.trim().replace(/^\|/u, '').replace(/\|$/u, '');
  return trimmed.split('|').map((cell) => cell.trim());
};

const isTableDivider = (line: string) => {
  const cells = splitTableRow(line);
  return cells.length > 0 && cells.every((cell) => tableDividerCellPattern.test(cell));
};

const startsBlock = (lines: string[], index: number) => {
  const line = lines[index] ?? '';
  if (!line.trim()) return true;
  if (fencedCodePattern.test(line) || headingPattern.test(line) || blockquotePattern.test(line)) return true;
  if (unorderedListPattern.test(line) || orderedListPattern.test(line)) return true;
  return line.includes('|') && isTableDivider(lines[index + 1] ?? '');
};

const renderHeading = (level: number, content: ReactNode[], key: string) => {
  switch (level) {
    case 1:
      return <h1 key={key}>{content}</h1>;
    case 2:
      return <h2 key={key}>{content}</h2>;
    case 3:
      return <h3 key={key}>{content}</h3>;
    case 4:
      return <h4 key={key}>{content}</h4>;
    case 5:
      return <h5 key={key}>{content}</h5>;
    default:
      return <h6 key={key}>{content}</h6>;
  }
};

export const renderAiChatMessageText = (text: string): ReactNode => {
  const lines = text.replaceAll('\r\n', '\n').split('\n');
  const blocks: ReactNode[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    const fence = line.match(fencedCodePattern);
    if (fence) {
      const language = fence[1] || undefined;
      const codeLines: string[] = [];
      index += 1;
      while (index < lines.length && !fencedCodePattern.test(lines[index])) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      blocks.push(
        <pre className="ai-chat-markdown-code-block" key={`code-${blocks.length}`}>
          <code data-language={language}>{codeLines.join('\n')}</code>
        </pre>,
      );
      continue;
    }

    const heading = line.match(headingPattern);
    if (heading) {
      blocks.push(
        renderHeading(
          heading[1].length,
          renderInlineMarkdown(heading[2], `heading-${blocks.length}`),
          `heading-${blocks.length}`,
        ),
      );
      index += 1;
      continue;
    }

    if (line.includes('|') && isTableDivider(lines[index + 1] ?? '')) {
      const headers = splitTableRow(line);
      const rows: string[][] = [];
      index += 2;
      while (index < lines.length && lines[index].includes('|') && lines[index].trim()) {
        rows.push(splitTableRow(lines[index]));
        index += 1;
      }
      blocks.push(
        <div className="ai-chat-markdown-table-wrap" key={`table-${blocks.length}`}>
          <table>
            <thead>
              <tr>
                {headers.map((cell, cellIndex) => (
                  <th key={`head-${cellIndex}`}>{renderInlineMarkdown(cell, `table-head-${cellIndex}`)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={`row-${rowIndex}`}>
                  {headers.map((_, cellIndex) => (
                    <td key={`cell-${cellIndex}`}>
                      {renderInlineMarkdown(row[cellIndex] ?? '', `table-${rowIndex}-${cellIndex}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }

    const quote = line.match(blockquotePattern);
    if (quote) {
      const quoteLines: string[] = [quote[1]];
      index += 1;
      while (index < lines.length) {
        const nextQuote = lines[index].match(blockquotePattern);
        if (!nextQuote) break;
        quoteLines.push(nextQuote[1]);
        index += 1;
      }
      blocks.push(
        <blockquote key={`quote-${blocks.length}`}>
          {renderInlineWithBreaks(quoteLines.join('\n'), `quote-${blocks.length}`)}
        </blockquote>,
      );
      continue;
    }

    const unordered = line.match(unorderedListPattern);
    const ordered = line.match(orderedListPattern);
    if (unordered || ordered) {
      const orderedList = Boolean(ordered);
      const items: string[] = [];
      const pattern = orderedList ? orderedListPattern : unorderedListPattern;
      while (index < lines.length) {
        const item = lines[index].match(pattern);
        if (!item) break;
        items.push(item[1]);
        index += 1;
      }
      const children = items.map((item, itemIndex) => (
        <li key={`item-${itemIndex}`}>{renderInlineMarkdown(item, `list-${blocks.length}-${itemIndex}`)}</li>
      ));
      blocks.push(
        orderedList ? <ol key={`list-${blocks.length}`}>{children}</ol> : <ul key={`list-${blocks.length}`}>{children}</ul>,
      );
      continue;
    }

    const paragraphLines = [line];
    index += 1;
    while (index < lines.length && !startsBlock(lines, index)) {
      paragraphLines.push(lines[index]);
      index += 1;
    }
    blocks.push(
      <p key={`paragraph-${blocks.length}`}>
        {renderInlineWithBreaks(paragraphLines.join('\n'), `paragraph-${blocks.length}`)}
      </p>,
    );
  }

  return <div className="ai-chat-markdown">{blocks}</div>;
};
