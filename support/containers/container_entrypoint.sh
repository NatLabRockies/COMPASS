#!/bin/bash
set -e

source "$(dirname -- "${BASH_SOURCE[0]}")/shell-hook"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME:?HOME must be set}/.cache}"
mkdir -p "$XDG_CACHE_HOME"

# Camoufox discovers its pre-fetched browser through the user cache.
if [ ! -e "$XDG_CACHE_HOME/camoufox" ] && \
   [ ! -L "$XDG_CACHE_HOME/camoufox" ]; then
    ln -s /opt/assets/cache/camoufox "$XDG_CACHE_HOME/camoufox"
fi

exec "$@"
