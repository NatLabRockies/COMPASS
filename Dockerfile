# syntax=docker/dockerfile:1

FROM ghcr.io/prefix-dev/pixi:0.49.0 AS build

ARG PIXI_ENV=default

COPY . /app
WORKDIR /app
VOLUME /app/outputs

RUN apt-get update && \
		apt-get install -y --no-install-recommends \
		build-essential \
		git && \
		rm -rf /var/lib/apt/lists/*

RUN pixi install --frozen -e ${PIXI_ENV}

RUN pixi run -e ${PIXI_ENV} playwright install

RUN pixi shell-hook -e ${PIXI_ENV} -s bash > /shell-hook
RUN echo "#!/bin/bash" > /app/entrypoint.sh
RUN cat /shell-hook >> /app/entrypoint.sh
RUN echo 'exec "$@"' >> /app/entrypoint.sh

FROM ubuntu:24.04 AS production

COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build /app/compass /app/compass
COPY --from=build /app/run.sh /app/run.sh
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh
WORKDIR /app

ENTRYPOINT [ "/app/entrypoint.sh" ]

CMD ["/bin/bash", "run.sh"]
