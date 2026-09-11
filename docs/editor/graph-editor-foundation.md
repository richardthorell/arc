# Shared Graph Editor Foundation

This document defines the first extraction step for ARC's reusable node graph editor.

## Goal

Provide domain-neutral graph UI/model primitives that the existing Material Graph can adopt without changing material compilation semantics. The same primitives must also be suitable for future Flow, Animation, and Behavior graphs.

## Shared responsibilities

The shared graph layer owns graph interaction and presentation mechanics:

- canvas pan/zoom and grid presentation
- node positioning and selection
- port presentation and connection gestures
- edge rendering
- marquee/multi-selection
- delete/duplicate/copy/paste interaction hooks
- context-menu and palette integration points
- viewport persistence hooks
- graph diagnostics presentation hooks

The shared layer must not encode Material-specific node types, shader semantics, or compilation behavior.

## Domain boundary

Each graph domain supplies its own node catalog, port rules, validation, inspector model, and presentation metadata through a `GraphDomain` contract. Shared components should not branch on concrete graph types.

The initial implementation intentionally focuses on reusable primitives and types. Migrating the existing Material Graph onto the new layer is the next milestone so that the abstraction can be validated against a real graph before Flow is introduced.
