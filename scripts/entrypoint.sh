#!/bin/sh
# Seed the cache volume from the copy baked into the image, once.
#
# The repository carries real SPHEREx pixels for the demo targets. Without this, a fresh machine would start empty and
# fetch every frame live the first time anyone opened it, which is exactly when a judge is watching.
set -e

CACHE="${QH_CACHE_DIR:-/data/cache}"
SEED="/srv/seed-cache"

if [ -d "$SEED" ] && [ ! -f "$CACHE/.seeded" ]; then
  mkdir -p "$CACHE"
  cp -R "$SEED"/. "$CACHE"/ 2>/dev/null || true
  touch "$CACHE/.seeded"
  echo "cache seeded from the image into $CACHE"
fi

chown -R qh "$CACHE"
export HOME=/home/qh
exec setpriv --reuid=qh --regid=qh --init-groups "$@"
