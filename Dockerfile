# Reproducible environment for drone_follow: the toy simulator, the full test
# suite, and a webcam dry run all run the same way here as on any machine --
# no more "works on my Mac" wheel/compiler surprises (see requirements-core.txt
# for why opencv-python is pinned the way it is).
#
# Build (fast, core only -- everything the test suite needs):
#   docker build -t drone_follow .
#
# Build with the optional insightface/MAVLink/serial extras too:
#   docker build --build-arg EXTRAS=full -t drone_follow:full .
#
# Run the test suite:
#   docker run --rm drone_follow python -m unittest discover -s tests -t . -v
#
# Run the toy simulator headless for a bit:
#   docker run --rm drone_follow python main.py --platform sim --seconds 60
#
# A real webcam dry run needs the host's camera passed through, which Docker
# can't do portably across platforms -- run that natively (see README), not
# in this container.

FROM python:3.13-slim

# opencv (even the "headless" build) dynamically links against a few system
# libraries that python:3.13-slim doesn't ship by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so `docker build` only reinstalls them when the
# requirements files actually change, not on every source edit.
COPY requirements-core.txt requirements-full.txt ./
RUN pip install --no-cache-dir --only-binary=:all: -r requirements-core.txt

ARG EXTRAS=core
RUN if [ "$EXTRAS" = "full" ]; then pip install --no-cache-dir -r requirements-full.txt; fi

COPY . .

# Build-time smoke test: an image that built successfully is one whose test
# suite actually passes, not just one whose dependencies happened to install.
RUN python -m unittest discover -s tests -t . -q

CMD ["python3", "main.py", "--platform", "sim", "--seconds", "30"]
