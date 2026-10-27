# install.sh --with-model, one real run (2026-10-19)

realrun.log: install.sh from branch alpha/ceiling-and-model run twice on the project Spark with
real downloads from huggingface.co (Qwen/Qwen3.8-27B-FP8 at 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a)
into a test folder. Setup: the folder was a hard-link copy of our checked model folder with
config.json and generation_config.json removed and tokenizer_config.json present only as its
first 4000 bytes in tokenizer_config.json.part. cargo was a stand-in (no build), so only the
model step is real.

- Run 1: free disk check passed; fetched config.json and generation_config.json, resumed
  tokenizer_config.json; sha256sum -c: all 81 files match; the three files are byte-identical to
  our copy (cmp). This run used install.sh before the banner fix.
- Run 2 (after the banner fix): 0 files fetched, all 81 match; the closing banner says the model
  was downloaded and checked.

Not done: a full 31 GB download into an empty folder.
