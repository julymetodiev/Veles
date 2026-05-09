# Installing Veles

Pre-built binaries are published for every tagged release. Pick whichever
fits your platform.

## Linux / macOS — one-liner

```sh
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/julymetodiev/Veles/releases/latest/download/veles-cli-installer.sh | sh
```

The script downloads the right binary for your `uname -m`, drops it into
`~/.cargo/bin/` (or whatever `$CARGO_HOME` points at), and prints the
directory you might need to add to `$PATH`.

## Windows — PowerShell one-liner

```powershell
irm https://github.com/julymetodiev/Veles/releases/latest/download/veles-cli-installer.ps1 | iex
```

## Homebrew

```sh
brew install julymetodiev/veles/veles-cli
```

The first command taps `julymetodiev/homebrew-veles` automatically. Same
formula is updated on every tag.

## From crates.io

```sh
cargo install veles-cli
```

Builds from source. Slower than the prebuilt binaries (tree-sitter pulls
in C compilation) but works on any platform Rust supports.

## Manual download

Browse the [Releases page](https://github.com/julymetodiev/Veles/releases),
grab the `.tar.xz` (Linux/macOS) or `.zip` (Windows) for your target,
extract, and place `veles` somewhere on `$PATH`.

Each archive ships with a SHA-256 checksum file; verify before extracting:

```sh
shasum -a 256 -c veles-cli-x86_64-unknown-linux-musl.tar.xz.sha256
```

## After install

Run `veles --version` to confirm. The first `search` call will download the
embedding model from Hugging Face into `~/.cache/huggingface/hub/`
(~64 MB, one-time). Subsequent runs reuse it.

For the full reference see [USAGE.md](USAGE.md).
