# Live Play L1.3b2a — Native scene-baseline outcome gate

Tracking: [#1198](https://github.com/richardthorell/arc/issues/1198),
[#1194](https://github.com/richardthorell/arc/issues/1194), and
[#1215](https://github.com/richardthorell/arc/pull/1215).

The L1.3b1 protocol can offer an immutable *already target-cooked* scene
generation and receive an authenticated, correlated activation result.
Until a C9-verified ephemeral release can be published and mounted, the
current Windows Development game previously accepted this frame at the
native transport but silently ignored it.

## This slice

- The Windows game network worker now explicitly rejects the
  `scene.baseline.offer` as `unavailable`, never claiming `active`.
  This does not modify ECS or attempt raw source delivery from the socket.
- The editor-owned `WindowsLivePlaySession.awaitBaselineStatus()` accepts
  the offer metadata **only from future trusted main-process C9 orchestration**,
  awaits the exact native result (the control server validates correlation),
  and distinguishes connected from scene-activated. Rejected, disconnected,
  crashed, stopped, timed-out or failed offers do not complete as ready.
- One outstanding attempt per runtime session; a later attempt may proceed
  after a completed or rejected result. Session Stop cancels the waiter and
  prevents stale session acknowledgments from completing a new session.
- No change to default Editor Viewport Play or published Shipping runtime.

## Remaining L1.3b

C9 #1137/#995 must materialize the session-owned L1.3a authored snapshot
as a compatible **cooked**, integrity-checked generation. C9 #1135 must
resolve it on the native runtime, followed by safe startup scene
activation via #1094. Only once that work is in place should the native
network worker dispatch the offer to the content/scene owner and report
`active` after *actual* activation.

**This is a readiness/negative-ACK integration slice, not scene streaming.**
It deliberately does not offer or transmit the raw `.arcscene`, cook
source files itself, expose an unsafe URL, or claim external Play reflects
unsaved changes yet.
