# Open-Source Attribution & Acknowledgments

## Modular (MAX Engine and Mojo), formerly used

Until 2026-10-15 this project used Modular MAX for model serving and comparison runs and Mojo for some GPU kernels, both by **Modular Inc.** (https://modular.com). They were removed on that date: no Modular code, library or build step is part of the current tree. Dated records under `docs/` and evidence folders still describe runs made with them. Modular, MAX and Mojo are trademarks of Modular Inc.

---

## Upstream Foundations

- **NVIDIA**: Blackwell GB10 architecture, TensorRT, NeMo/Nemotron models, and CUDA driver runtime.
- **The Rust Community**: Tokio, Axum, Ort (ONNX Runtime C-API bindings), and HuggingFace Tokenizers.
- **Radicle**: Sovereign peer-to-peer code collaboration and Heartwood git protocol.

---

## Third-Party Model: SmolLM2-1.7B-Instruct (Hugging Face)

The runtime can load and run **SmolLM2-1.7B-Instruct** by Hugging Face (HuggingFaceTB),
an open-weights instruction model. It is a separate third-party component, not part of
the AIEN source code, and it keeps its own licence.

- **Source**: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct, revision
  `31b70e2e869a7173562077fd711b654946d38674`. Weights `model.safetensors` sha256
  `f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8` (equal to the
  publisher's Git LFS object id). Used unmodified: no conversion, no quantization.
- **Licence**: Apache License 2.0 (`Apache-2.0`), as declared in the model card
  (`license: apache-2.0`). Full text: `LICENSES/Apache-2.0.txt`. The repository at this
  revision ships no separate LICENSE or NOTICE file; its model card (`README.md`) is the
  publisher's attribution and is distributed unchanged with the weights.
- **Attribution** (the model card's citation): Loubna Ben Allal, Anton Lozhkov, Elie
  Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti,
  Hynek Kydlíček, Agustín Piqueres Lajarín, Vaibhav Srivastav, Joshua Lochner, Caleb
  Fahlgren, Xuan-Son Nguyen, Clémentine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun
  Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra, Thomas Wolf,
  "SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model",
  arXiv:2502.02737 (2025).
- **Boundary**: AIEN code stays under Apache-2.0. The model files stay under
  Apache-2.0 with their notices preserved. Running the model through AIEN, or storing it
  under another directory or file name, does not change either licence.
