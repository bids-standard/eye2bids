# syntax=docker/dockerfile:1

ARG UBUNTU_VERSION=26.04

# ---------------------------------------------------------------------------
# Builder: build the eye2bids wheel and install it (runtime deps only) into a
# virtualenv. Nothing from this stage except the venv ends up in the image.
# ---------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git is required by hatch-vcs to derive the package version from the repo.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /src
COPY . .

# The checkout is owned by another uid than root, so git refuses to read it
# without this; hatch-vcs would then fail to determine the version.
RUN git config --global --add safe.directory /src \
    && pip install .

# ---------------------------------------------------------------------------
# Runtime: minimal image with the venv, the EyeLink tools (edf2asc is invoked
# by eye2bids) and an unprivileged user.
# ---------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

# Install the SR Research apt repository and eyelink-display-software (edf2asc).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        python3 \
    && install -d -m 0755 /etc/apt/keyrings \
    && curl -fsSL https://apt.sr-research.com/SRResearch_key.gpg \
        | gpg --dearmor -o /etc/apt/keyrings/SRResearch_key.gpg \
    && curl -fsSL https://apt.sr-research.com/SR_Research_repo.sources \
        -o /etc/apt/sources.list.d/SR_Research_repo.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends eyelink-display-software \
    && rm -rf /var/lib/apt/lists/*

# The venv is created in the builder at /opt/venv, so it must be copied to the
# same path (its scripts reference that absolute path).
COPY --from=builder /opt/venv /opt/venv

# Unprivileged user. The ubuntu base image already ships a user with uid 1000.
RUN useradd --create-home --uid 10001 --shell /bin/bash eye2bids
USER eye2bids

# Input and output are expected to be bind-mounted here.
WORKDIR /data

ENTRYPOINT ["eye2bids"]
CMD ["--help"]
