<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# Third-party data notices

This notice applies to the adapted public prompts in
`data/train/inputs-base.json` and `data/dev/inputs-base.json`. Those files are
collections. Each source-derived part retains the license below; the project
Apache-2.0 license does not relicense third-party material. Exact revisions,
artifact hashes, and license-evidence hashes are in
`data/sources/source-pins.v1.json`.

## Belebele Korean

Source: Meta's Belebele `kor_Hang` configuration. Licensed under
[CC BY-SA 4.0](LICENSES/CC-BY-SA-4.0.txt). The released adaptation selects a
subset, formats passage, question, and choices as a prompt, omits answer
labels, and assigns opaque episode IDs. No endorsement is implied.

Attribution: Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe,
Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke
Zettlemoyer, and Madian Khabsa, *The Belebele Benchmark: a Parallel Reading
Comprehension Dataset in 122 Language Variants*, ACL 2024.

## CRUXEval

Copyright (c) 2023 Meta. Licensed under the [MIT License](LICENSES/MIT.txt).
The adaptation selects public examples, applies the direct input-prediction or
output-prediction prompt, omits reference inputs or outputs, and assigns opaque
episode IDs.

## GSM8K

Copyright (c) 2021 OpenAI. Licensed under the
[MIT License](LICENSES/MIT.txt). The adaptation selects public test questions,
omits solutions and answers, and assigns opaque episode IDs.

## BABILong 4K/16K components

The bAbI tasks component is copyright (c) 2015-present Facebook, Inc. and is
licensed under [BSD-3-Clause](LICENSES/BSD-3-Clause.txt). BABILong code and the
PG-19 component are licensed under [Apache-2.0](LICENSES/Apache-2.0.txt).
The adaptation uses only approved 4K and 16K configurations, adds a zero-shot task
instruction, omits targets, and assigns opaque episode IDs. Neither Facebook
nor any contributor endorses this project.

## Apache-2.0 sources

The following adapted public prompts are licensed under
[Apache-2.0](LICENSES/Apache-2.0.txt): DeepMind Mathematics, HRMCR, RuleTaker,
and TruthfulQA. Each adaptation selects an approved subset, formats only the
question-side prompt, omits gold answers and solutions, and assigns opaque
episode IDs. DeepMind Mathematics prompts are independently reproduced from
the pinned upstream generator and verified against the reference hashes in the
source record.

## Source-fetch-only material

AIME problem text is not included in this repository or release archive.
`data/train/aime-selection.json` and `data/dev/aime-selection.json` contain
only public source keys and expected prompt hashes. Users fetch the pinned
public sources and materialize those prompts locally.

## Bundled sentence encoder (multilingual-e5-small, ONNX int8)

`src/ossp_router/resources/e5-small-int8.onnx.part00`/`.part01` (joined at
image build to `e5-small-int8.onnx`, SHA-256
`4d24e2bc01a447951524466ef533e52944bf48509e6552810bcee1a2711cb02c`) and
`src/ossp_router/resources/e5-tokenizer.json` (SHA-256
`0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39`) are the
int8-quantized ONNX export of `intfloat/multilingual-e5-small`
(Wang et al., "Text Embeddings by Weakly-Supervised Contrastive Pre-training",
2022), obtained from `https://huggingface.co/Xenova/multilingual-e5-small`
at pinned revision `761b726dd34fb83930e26aab4e9ac3899aa1fa78`
(files `onnx/model_int8.onnx`, `tokenizer.json`). Both the original model and
the ONNX conversion are licensed under the
[MIT License](LICENSES/MIT.txt). The encoder is used offline inside the
router container as a fixed feature extractor over the current prompt only;
no third-party data beyond the model weights is included.

The runtime container installs `numpy` (BSD-3-Clause), `onnxruntime` (MIT)
and `tokenizers` (Apache-2.0) from PyPI at image build time, pinned in
`container/Dockerfile`; each wheel retains its upstream LICENSE (and any
NOTICE) files under `site-packages` inside the image. The base image
`python:3.11-slim-bookworm` carries Debian operating-system components
whose per-package copyright and license texts remain in the image under
`/usr/share/doc/*/copyright`; corresponding sources are published by the
Debian project.
