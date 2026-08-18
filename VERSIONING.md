# Versioning Policy

`ovannotators` follows [Semantic Versioning 2.0.0](https://semver.org/).

## Version format

All published versions use `MAJOR.MINOR.PATCH`.

## What each increment means

- `MAJOR`: backward-incompatible changes to public APIs or published contracts.
- `MINOR`: backward-compatible feature additions.
- `PATCH`: backward-compatible bug fixes or implementation corrections.

## Scope of this policy

Semantic versioning applies to:

- The `ovannotators` package version.
- The published writer surface (writer registry, custom writer extension API,
  payload transforms, run record format).
- The consumer side of the input payload contract with the renderer (the
  AOV/buffer payload, anchored to the `ovrtx` self-describing producer payload).
- The consumer side of the upstream randomized-scene handoff (the variant scene
  reference and the producer run-record reference), which does not carry
  AOV/buffer payloads.

## Compatibility expectations

- Consumers should treat `MAJOR` changes as requiring explicit migration.
- `MINOR` and `PATCH` releases are expected to remain backward-compatible
  within the same major series.
- Release notes must call out compatibility-impacting changes and any required
  migration steps.

## Pre-1.0 note

Before `1.0.0`, breaking changes may happen more frequently. They are still
documented in release notes and reflected in version increments.
