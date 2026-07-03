# GLM-5.2-594B

GLM-5.2 NVFP4-REAP-594B: vLLM serving on 4× RTX PRO 6000 (Blackwell, sm120, TP4)
Measured single-stream: ~80 t/s codegen, ~240k KV context.

DCP4 + MTP5 + FLASHINFER_MLA_SPARSE_SM120 + b12x MoE

## Running

Retrieve the model, cache it locally, and then run the docker-compose file:

```
# Specify some dir that you're going to be storing models in.
% export MODEL_DIR=~/storage/models  

% export MODEL_NAME=GLM-5.2-Int8Mix-NVFP4-REAP-594B

# You probably already have this installed...
% pip install huggingface_hub

% hf download madeby561/${MODEL_NAME} --local-dir ${MODEL_DIR}/${MODEL_NAME}

# Run this docker config
% docker compose up
```
