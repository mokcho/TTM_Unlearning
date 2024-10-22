#!/bin/bash

# TODO: change the yaml, text list, ckpt path

python3 infer_mos5.py \
--config_yaml audioldm_train/config/mos_as_token/qa_mdt.yaml \
--list_inference prompts/good_prompts_1.lst \
--reload_from_ckpt "output/model_checkpoint.ckpt"