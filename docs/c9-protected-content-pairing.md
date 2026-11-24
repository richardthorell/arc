# C9.4c — Authenticated portable cooked-content transport

Tracking: [#1137](https://github.com/richardthorell/arc/issues/1137) and
[C9 #1133](https://github.com/richardthorell/arc/issues/1133).

## Integration with Live Play L1.3

Merged [#1247](https://github.com/richardthorell/arc/pull/1247) now owns
the protected Windows native launch handoff, per-session C9 credential,
authenticated WinHTTP discovery and **verified remote scene activation**.
Do not duplicate its native pairing schema, editor main-process orchestration,
Windows entrypoint, or activation gates.

This PR provides the previously missing **portable transport parity**:
the C9 HTTP contract stores a Development credential in
`content_http_limits::development_bearer_token`, already validated by
the shared C9 request policy. This PR uses that same setting in native
libcurl and Android's JNI/platform Java HTTP adapter for both immutable cooked
objects (including ranges) and mutable current.json release lookups.

The Authorization header is set only after request preflight. A paired credential
requires explicit Development HTTP to 127.0.0.1 (port validated by arc-http);
it must not be sent to LAN origins, HTTPS hosts or redirect destinations.
The existing certificate/hostname verification, redirect denial, bounded
transfer size and no-cache release semantics remain unchanged.

The Java adapter retains its existing five-argument anonymous `open`
overload for compatibility with current Android hosts and Java tests.
The paired overload adds the session credential without changing request URLs,
Intent extras, argv, files or logging.

## Limitations

This does **not** launch Android Live Play or install ADB reverse forwarding;
that still requires a device-specific, trusted pairing and lifecycle follow-up
(C9.4c2). No LAN bind, persistent credential, cross-process token registry
or remote-world activation is added here. Windows native activation already
belongs to #1247. C9.4d still owns generation change-feed/reconnect recovery.

## Verification

- Java SE loopback host test checks anonymous request, authorized release GET,
  malformed credential rejection, off-host pairing denial, and existing
  ranges/ETags/redirect/encoding restrictions.
- Existing shared arc-http tests cover valid and malformed
  `development_bearer_token`, origin requirements and target-scoped release
  requests. The native CI matrix builds the updated libcurl and Android bridges.
