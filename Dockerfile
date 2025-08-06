# syntax=docker/dockerfile:1

FROM ghcr.io/prefix-dev/pixi:0.50.2 AS build

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
#     dist/nrel_compass-*.whl
# Do we want to remove pip to reduce production's size?
# RUN pixi remove --frozen -e ${PIXI_ENV} pip


RUN pixi shell-hook -e ${PIXI_ENV} -s bash > /shell-hook
RUN echo "#!/bin/bash" > /app/entrypoint.sh
RUN cat /shell-hook >> /app/entrypoint.sh
RUN echo 'exec "$@"' >> /app/entrypoint.sh

# ==== Production ====
FROM ubuntu:24.04 AS production

COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
# Consider installing from wheels in the future so that we don't need to copy the source
COPY --from=build /app/compass /app/compass
COPY --from=build /app/run.sh /app/run.sh
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh
WORKDIR /app

# Get browser binaries
RUN /app/.pixi/envs/default/bin/playwright install --with-deps
RUN /app/.pixi/envs/default/bin/rebrowser_playwright install --with-deps
RUN /app/.pixi/envs/default/bin/camoufox fetch

ENTRYPOINT [ "/app/entrypoint.sh" ]

CMD ["/bin/bash", "run.sh"]
