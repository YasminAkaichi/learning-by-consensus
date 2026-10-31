# Learning by Consensus – Popper v4 + Andante + Flower, ready to run.
# SWI-Prolog (needed by Popper through janus-swi) comes from the official image.
FROM swipl:stable

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Results are written here: mount a folder to keep them (-v "$PWD/results:/app/results").
VOLUME ["/app/results"]

ENTRYPOINT ["python", "-m", "filp_consensus"]
CMD ["run", "--dataset", "zendo1", "--learner", "popper", "--clients", "3", "--partition", "iid", "--seed", "42", "--show-predictions"]
