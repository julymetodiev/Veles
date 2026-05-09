# Releasing Veles

Releases are driven by [cargo-dist](https://opensource.axo.dev/cargo-dist/).
Pushing a `v*` tag triggers `.github/workflows/release.yml`, which builds
binaries for every target listed in `dist-workspace.toml`, uploads them to
a GitHub Release, updates the Homebrew tap, and finally pushes the four
crates to crates.io.

## One-time setup

1. **Create the Homebrew tap repo** on GitHub:
   `julymetodiev/homebrew-veles` — empty, public, default branch `main`.
   No files needed; cargo-dist will populate it.

2. **Add repository secrets** at
   `https://github.com/julymetodiev/Veles/settings/secrets/actions`:

   | Secret | Source | Used for |
   |---|---|---|
   | `CARGO_REGISTRY_TOKEN` | crates.io → Account → API Tokens, scope `publish-update` | `publish-crates` workflow |
   | `HOMEBREW_TAP_TOKEN` | GitHub → Settings → Developer settings → PAT (fine-grained), scope `Contents: Read and write` on `homebrew-veles` | cargo-dist Homebrew job |

   The `HOMEBREW_TAP_TOKEN` is required because GitHub's default
   `GITHUB_TOKEN` can only push to the repo it ran in. cargo-dist needs to
   commit to the tap repo, which is separate.

3. **Publish each crate manually for the first time** (cargo-dist's
   publish-crates workflow uses `cargo publish` against an existing crate;
   the very first publish has to come from a maintainer's local machine):

   ```sh
   cargo login                    # paste your crates.io token
   cargo publish -p veles-core
   cargo publish -p veles-grpc
   cargo publish -p veles-mcp
   cargo publish -p veles-cli
   ```

   After this, the workflow handles every subsequent release.

## Cutting a release

1. Make sure `main` has everything you want and CI is green.

2. Bump the version in `Cargo.toml`'s `[workspace.package]`:

   ```toml
   [workspace.package]
   version = "0.2.0"
   ```

3. Commit:

   ```sh
   git add Cargo.toml Cargo.lock
   git commit -m "release: v0.2.0"
   ```

4. Tag and push:

   ```sh
   git tag v0.2.0
   git push origin main
   git push origin v0.2.0
   ```

5. Watch the **Release** workflow at
   `https://github.com/julymetodiev/Veles/actions`. It runs:

   1. `plan`        — validate config
   2. `build-*`     — one job per target triple (~3-5 min each, parallel)
   3. `host`        — collects artifacts, generates checksums + installers
   4. `publish-homebrew` — pushes the formula update to `homebrew-veles`
   5. `publish-crates`   — runs `cargo publish` for the four crates
   6. `announce`    — creates the GitHub Release page

   If `publish-crates` fails (network, propagation timing, name conflict),
   the GitHub Release and Homebrew formula are still good — you can re-run
   only that job from the Actions UI.

## Pre-releases

Tags with a hyphen (`v0.2.0-rc.1`, `v0.2.0-beta`, …) are marked as
pre-releases automatically and are not pushed to crates.io / Homebrew by
default. Useful for testing the binaries before a stable cut.

## Updating cargo-dist itself

```sh
cargo install cargo-dist
dist init --yes        # bumps cargo-dist-version + regenerates release.yml
git diff .github/workflows/release.yml dist-workspace.toml
```

Review the diff, commit, ship.
