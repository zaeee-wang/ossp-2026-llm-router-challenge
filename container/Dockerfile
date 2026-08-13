# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# Debian slim instead of alpine: the router embeds prompts with a bundled ONNX
# encoder, and onnxruntime publishes no musl wheels. The size limits (1 GiB
# compressed layers, 2 GiB merged root) leave ample room.
FROM python:3.11.15-slim-bookworm

# Inference dependencies, pinned. Network is used at BUILD time only; the
# running container is offline and imports these from site-packages.
RUN pip install --no-cache-dir \
        numpy==2.0.2 \
        onnxruntime==1.28.0 \
        tokenizers==0.23.1 \
    && pip uninstall --yes pip setuptools wheel

ARG SOURCE_MANIFEST_SHA256=unbound

LABEL io.sktelecom.ossp.source-manifest-sha256="${SOURCE_MANIFEST_SHA256}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/router \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp

COPY --chown=65532:65532 src container/entrypoint.py /opt/router/

# Git hosting caps single files below the encoder's 118 MB, so the repository
# carries it split; join it here and verify the digest of the reassembled file.
RUN cd /opt/router/ossp_router/resources \
    && cat e5-small-int8.onnx.part* > e5-small-int8.onnx \
    && echo "4d24e2bc01a447951524466ef533e52944bf48509e6552810bcee1a2711cb02c  e5-small-int8.onnx" | sha256sum -c - \
    && rm e5-small-int8.onnx.part* \
    && chown 65532:65532 e5-small-int8.onnx

WORKDIR /opt/router

USER 65532:65532

ENTRYPOINT ["python3", "/opt/router/entrypoint.py"]
