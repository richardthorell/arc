# Asset library architecture

ARC should treat **where an asset is owned** separately from **where an asset was discovered**.

## Storage scopes

| Scope | Purpose | Expected ownership |
| --- | --- | --- |
| Built-in | Engine-shipped primitives, materials, templates and editor resources | ARC installation, read-only |
| Project | Assets committed or otherwise pinned to one project | Project/team |
| User | A private reusable library available to the signed-in user across projects | User |
| Organization | Shared studio/team library with permissions and versions | Organization |

Favorites, Recent, Downloads and Search Results should be virtual views rather than additional storage scopes.

The browser should present these scopes as mount points instead of encoding them into arbitrary filesystem paths. Stable asset IDs should remain authoritative; paths are display and organization metadata.

Suggested logical URIs:

- `asset://builtin/...`
- `asset://project/...`
- `asset://user/...`
- `asset://org/<organization-id>/...`

## Search metadata and facets

The Content Browser search model is layout-independent. Search matches asset name, title, description, path and tags, and composes with folder-prefix, asset-type, tag and state filters. Type and tag facets are derived deterministically from the matching metadata population so the UI can show useful counts without encoding view state into asset metadata.

Project-owned searchable metadata is stored in the versioned project file `.arc-asset-metadata.json`. Entries prefer the stable asset GUID as identity and fall back to a normalized asset path only when no GUID exists. The file stores only authoring metadata (`title`, `description`, and `tags`); grid/list selection, folder expansion, sort order and other Content Browser layout state remain editor UI state.

Built-in/read-only assets expose their registry metadata for search and inspection but cannot be edited from the project metadata dialog. Type-specific details such as texture resolution/format, model geometry counts, material references, shader stages, prefab composition and scene counts continue to come from the asset registry rather than being duplicated into the metadata file.

## Structured search grammar

Free text and structured predicates can be mixed in one query. Free-text terms are sent to the lexical/semantic ranking path; predicates constrain the candidate set before ranking.

Initial S2 predicates:

| Predicate | Meaning | Example |
| --- | --- | --- |
| `type:` | Asset type alias or stable type UUID | `type:mesh`, `type:material` |
| `tag:` | Required authored or generated tag | `tag:rustic` |
| `category:` | Required category ID/path | `category:"Furniture/Chair"` |
| `provider:` | Exact provider ID or common provider alias | `provider:arc.project.source` |
| `scope:` | Common logical source grouping | `scope:project`, `scope:builtin` |
| typed property | Numeric/string asset property | `triangles:<100000`, `roughness:>0.8`, `format:bc7` |

Examples:

```text
rustic chair type:mesh
chair tag:rustic tag:"old wood" category:"Furniture/Chair"
type:material scope:project
rustic chair type:mesh triangles:<100000
type:material roughness:>0.8
type:texture format:bc7
```

Predicates are case-insensitive where their stored vocabulary permits it. Repeated tag/category predicates use **AND** semantics. A scope may expand to multiple provider IDs internally; those providers form an OR-set within the scope filter.

Values containing whitespace can be quoted. Inside a quoted value, `\\"` and `\\` escape a quote and backslash respectively.

Unknown predicates, unknown type/scope aliases, conflicting `type:` predicates, missing values, and unterminated quoted values are parse errors. ARC should surface the parser's token/offset/message to the user instead of silently treating a malformed predicate as ordinary free text.


### Typed search properties

Search properties are typed, indexed derived metadata rather than free-form tags. Numeric properties support `=`, `<`, `<=`, `>`, and `>=`; string properties support case-insensitive equality.

Initial property vocabulary reserves:

- Mesh: `triangles`, `vertices`
- Texture: `width`, `height`, `format`, `compression`
- Material: `roughness`, `metallic`, `blend-mode`
- Audio/Sound: `duration`, `sample-rate`, `channels`, `encoding`
- Common: `dependencies`, `reverse-dependencies`, `generation`, `revision`

The search subsystem indexes and queries these values but should **not** open asset files to derive them. Asset/importer-specific code publishes authoritative properties into the search document/enrichment path. The common dependency/generation/revision properties are available directly from the Asset Database today; type-specific importer publication is a follow-up S2 slice.

## Online sources

Online libraries are **sources**, not scopes. A source adapter normalizes provider-specific catalogs into ARC's common source contract. The Content Browser can search one source or aggregate multiple sources, while keeping provider-specific networking and response formats outside the UI.

Examples:

- Poly Haven
- ambientCG
- studio-internal DAM or object storage
- future marketplace providers

When a remote asset is imported, ARC should choose a destination scope (Project by default), run the regular importer/cooker pipeline, and persist provenance alongside the imported asset:

- source adapter ID
- provider asset ID
- import timestamp
- source revision/hash when available
- license at import time
- original source URL
- selected download variant/import recipe

That makes re-imports deterministic and lets the editor answer where an asset came from even if the source catalog later changes.

## Adapter contract

`ArcAssetSourceAdapter` intentionally exposes only normalized operations:

1. `search` returns common asset metadata for browsing and filtering.
2. `getAsset` resolves one provider asset by stable provider ID.
3. `getDownloadManifest` returns downloadable files, sizes and checksums without deciding how ARC imports them.

Downloading, checksum verification, cache placement, conversion and project import should remain ARC services layered above the adapter.

## First provider: Poly Haven

The first adapter is Poly Haven. It maps HDRIs, textures and models into ARC remote asset kinds, caches catalog metadata briefly, records CC0 as the asset license, preserves the required `Powered by Poly Haven` attribution string, sends an ARC-specific User-Agent, and flattens provider download metadata into ARC's common download manifest.
