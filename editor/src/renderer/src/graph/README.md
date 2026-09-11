# Shared graph editor primitives

This directory owns graph-domain-neutral editor contracts and presentation primitives. Material, Flow, Animation, and other graph types provide their own node catalogs, validation rules, payload editors, compilation, and runtime semantics.

Shared graph code must not branch on a concrete graph kind. Domain-specific behavior belongs behind `GraphDomain` or in the graph type's own editor adapter.
