# Live Play L1.3b2c2 — Immutable C9 release assembly and cooked build identity

Follows [#1222](https://github.com/richardthorell/arc/pull/1222),
[#1218](https://github.com/richardthorell/arc/pull/1218) and issue
[#1198](https://github.com/richardthorell/arc/issues/1198).

## Runtime versus cooked-content build identity

The packaged Development runtime is bound to a stable `baseBuildId`, but
unsaved Authoring World bytes produce a new cook-manifest `buildId` for each
unique source state. Existing C9.1 and C9.2c2 required the two IDs to match,
which would reject every legitimate ephemeral scene generation.

C9 release v1 now accepts an **optional** `cookBuildId` alongside the
unchanged required `baseBuildId`. The runtime still validates project,
engine, base build, exact profile and target. The immutable cooked artifact
provider compares the native `.arccookmanifest` build ID with
`cookBuildId` if provided, otherwise with `baseBuildId`. Standard/legacy
releases remain unchanged. Invalid or unsafe `cookBuildId` values fail closed.
The C++ parser, publisher-side Python contract, and C9.2c2 mount tests cover
both cases.

## Session-owned immutable release inventory

`prepareLivePlayRemoteRelease` accepts a **previously verified** private
cook/package result from L1.3b2c1 and produces C9.1 release descriptors plus
content-hash-addressed immutable objects. It does not duplicate scene cooking.

- Validate each package's native cook manifest, target/profile, registered
  scene asset root, Development configuration, texture family, cooked build
  identity, and exact package chunk closure.
- Copy manifest and chunk files into the session's private immutable
  `objects/sha256/<hash>` tree, recompute their SHA-256 after copying, then
  verify native ARCPAK1 headers and every artifact's SHA-256, offset and
  non-overlapping bounds.
- Produce one complete `targets/<texture-family>/current.json` for each
  cooked variant using the **existing C9.1** release format, with deterministic
  metadata and a relative immutable object inventory.
- Build everything in a private candidate directory. On any error, cancellation
  or session mismatch, discard the candidate; publish by one directory rename
  only after all targets have passed verification.
- Return each descriptor's digest for the L1 baseline offer protocol. Neither
  source scene paths nor session authentication tokens appear in the
  release descriptor.

## Remaining integration

This is **not** a network content server. The C9 #1137 editor-hosted server
must serve this immutable inventory and authenticate its Development client
sessions; C9.3 #1136 must select that endpoint at runtime. After the runtime
has acquired, verified, mounted, and activated the chosen cooked scene, its
L1.3 control bridge may ACK `active`. The existing Play button is unchanged
until those pieces are ready. The project source and normal published cooked
outputs remain untouched. Track #1198 stays open.
