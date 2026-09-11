# Shared Graph Editor Foundation

This document defines ARC's reusable node graph editor boundary.

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

## Adoption status

G1 established the shared graph contracts, geometry, measurement helpers, and rendering primitives. G2 migrates the production Material Graph canvas onto those primitives and routes Material node definitions, connection validation, and protected-node rules through `materialGraphDomain`.

Material-specific value editors, palette taxonomy, document history, persistence, and compilation remain in the Material domain. That separation is intentional: the shared layer owns graph mechanics while each domain owns graph meaning.

With Material now exercising the shared layer end to end, Flow can build on the same graph foundation without introducing Material-specific branches into the reusable graph code.
