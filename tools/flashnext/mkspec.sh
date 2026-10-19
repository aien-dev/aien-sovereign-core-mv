#!/usr/bin/env bash
# writes spec files for the format study: layers 0 24 47, experts 0 311
d=$1; mkdir -p "$d"; : > "$d/bf16.spec"; : > "$d/fp8.spec"; : > "$d/nvfp4.spec"
for L in 0 24 47; do for E in 0 311; do
  P=model.language_model.layers.$L.mlp.experts
  echo "$P.$E.gate_up_proj $P.gate_up_proj $E 1" >> "$d/bf16.spec"
  echo "$P.$E.down_proj $P.down_proj $E 1" >> "$d/bf16.spec"
  for m in gate up down; do
    echo "$P.$E.${m}_proj.weight $P.$E.${m}_proj.weight" >> "$d/fp8.spec"
    echo "$P.$E.${m}_proj.weight_scale_inv $P.$E.${m}_proj.weight_scale_inv" >> "$d/fp8.spec"
    for s in weight weight_scale weight_scale_2 input_scale; do echo "$P.$E.${m}_proj.$s $P.$E.${m}_proj.$s" >> "$d/nvfp4.spec"; done
  done
done; done
