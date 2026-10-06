import { parseArcUri, type ArcUri } from './arcUri';

export type ArcResourceMetadata = {
  uri: ArcUri;
  label: string;
  subtitle?: string;
  generation?: number;
  metadata?: Record<string, unknown>;
};

export type ArcResourceData = {
  uri: ArcUri;
  mediaType: string;
  dataUrl?: string;
  text?: string;
  generation?: number;
  metadata?: Record<string, unknown>;
};

export type ArcResourceHandler = {
  kind: string;
  resolve: (uri: ArcUri) => ArcResourceMetadata | null | Promise<ArcResourceMetadata | null>;
  read?: (uri: ArcUri) => ArcResourceData | null | Promise<ArcResourceData | null>;
};

export class ArcResourceRegistry {
  private readonly handlers = new Map<string, ArcResourceHandler>();

  register(handler: ArcResourceHandler): () => void {
    if (!/^[a-z][a-z0-9-]*$/u.test(handler.kind)) throw new Error(`Invalid ARC resource kind: ${handler.kind}`);
    if (this.handlers.has(handler.kind)) throw new Error(`ARC resource handler already registered: ${handler.kind}`);
    this.handlers.set(handler.kind, handler);
    return () => {
      if (this.handlers.get(handler.kind) === handler) this.handlers.delete(handler.kind);
    };
  }

  async resolve(value: string | ArcUri): Promise<ArcResourceMetadata | null> {
    const uri = typeof value === 'string' ? parseArcUri(value) : value;
    if (!uri) return null;
    return (await this.handlers.get(uri.kind)?.resolve(uri)) ?? null;
  }

  async read(value: string | ArcUri): Promise<ArcResourceData | null> {
    const uri = typeof value === 'string' ? parseArcUri(value) : value;
    if (!uri) return null;
    const handler = this.handlers.get(uri.kind);
    if (!handler?.read) return null;
    return (await handler.read(uri)) ?? null;
  }

  has(kind: string): boolean {
    return this.handlers.has(kind);
  }
}
