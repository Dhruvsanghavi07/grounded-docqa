#!/usr/bin/env bash
# Download the FastAPI tutorial docs (MIT licensed) as the demo corpus.
# Pinned to one commit so the evaluation numbers in the README are reproducible.
set -euo pipefail
COMMIT="${FASTAPI_DOCS_COMMIT:-5cb6272ed7e3e07eaf6978e1e65b94aca76fa8e0}"
DEST="${1:-backend/data/corpus}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git init --quiet "$TMP/fastapi"
git -C "$TMP/fastapi" remote add origin https://github.com/fastapi/fastapi.git
git -C "$TMP/fastapi" sparse-checkout set docs/en/docs/tutorial LICENSE
git -C "$TMP/fastapi" fetch --quiet --depth 1 --filter=blob:none origin "$COMMIT"
git -C "$TMP/fastapi" checkout --quiet FETCH_HEAD

rm -rf "$DEST" && mkdir -p "$DEST"
cp -r "$TMP/fastapi/docs/en/docs/tutorial/." "$DEST/"
cp "$TMP/fastapi/LICENSE" "$DEST/LICENSE-fastapi-docs"
echo "fastapi docs commit: $COMMIT" > "$DEST/SOURCE.txt"
echo "corpus ready in $DEST ($(find "$DEST" -name '*.md' | wc -l) markdown files)"
