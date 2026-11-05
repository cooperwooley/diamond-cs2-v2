#!/usr/bin/env bash
# Build the demo exporter (tools/demodump, #72) into bin/: bin/demodump for this machine's OS and bin/demodump.exe for
# Windows (cross-compiled, pure Go, no C toolchain needed). Needs Go >= the version in tools/demodump/go.mod; set GO
# to use a Go that is not on PATH.
#
#   scripts/build_demodump.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
go_bin="${GO:-go}"
if ! command -v "$go_bin" >/dev/null 2>&1; then
    echo "build_demodump: Go not found (install it from https://go.dev/dl/ or set GO=/path/to/go)" >&2
    exit 1
fi

mkdir -p "$root/bin"
cd "$root/tools/demodump"
"$go_bin" vet ./...
"$go_bin" test ./...
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 "$go_bin" build -trimpath -o "$root/bin/demodump" .
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 "$go_bin" build -trimpath -o "$root/bin/demodump.exe" .
ls -l "$root/bin/demodump" "$root/bin/demodump.exe"
"$root/bin/demodump" --version
