# syntax=docker/dockerfile:1

FROM ghcr.io/prefix-dev/pixi:0.68.1 AS build

ARG PIXI_ENV=default

COPY . /app
WORKDIR /app
VOLUME /app/outputs

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        pkg-config \
        libssl-dev \
        build-essential \
        ca-certificates \
        git && \
    rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates

RUN pixi install --frozen -e ${PIXI_ENV}

RUN pixi run --frozen -e pbuild build-wheels
# RUN pixi add --frozen pip
# RUN pixi run --frozen -e ${PIXI_ENV} \
#     python3 -m pip install --no-deps --disable-pip-version-check \
#     dist/infra_compass-*.whl
# Do we want to remove pip to reduce production's size?
# RUN pixi remove --frozen -e ${PIXI_ENV} pip


RUN pixi shell-hook -e ${PIXI_ENV} -s bash > /shell-hook

# ==== Production ====
FROM ubuntu:24.04 AS production

COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
# Consider installing from wheels in the future so that we don't need to copy the source
COPY --from=build /app/compass /app/compass
COPY --from=build /app/run.sh /app/run.sh
COPY --from=build /shell-hook /app/shell-hook
COPY --chmod=0755 support/containers/container_entrypoint.sh /app/entrypoint.sh
WORKDIR /app

# Get all necessary binaries
ENV PATH="/app/.pixi/envs/default/bin:${PATH}" \
    PLAYWRIGHT_BROWSERS_PATH=/opt/assets/playwright \
    PLAYWRIGHT_SKIP_BROWSER_GC=1 \
    DOCLING_RS_CACHE_DIR=/opt/assets/docling-rs \
    NLTK_DATA=/opt/assets/nltk \
    COMPASS_FILE_LOAD_BACKEND=docling

RUN export XDG_CACHE_HOME=/opt/assets/cache && \
    mkdir -p /opt/assets && \
    python -m playwright install --with-deps && \
    python -m rebrowser_playwright install --with-deps && \
    python -m patchright install --with-deps chromium && \
    scrapling install --force && \
    python -m camoufox fetch --browserforge && \
    crawl4ai-setup && \
    python -c "import docling_rs; docling_rs.download_models()" && \
    python -c "import nltk; [nltk.download(resource, download_dir='/opt/assets/nltk', raise_on_error=True) for resource in ('punkt', 'punkt_tab', 'stopwords')]" && \
    chmod -R a+rX /opt/assets

ENTRYPOINT [ "/app/entrypoint.sh" ]

CMD ["/bin/bash", "run.sh"]
