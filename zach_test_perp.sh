#!/bin/bash

# Baseline
python main.py \
	  mode=sample_eval \
	  seed=1234 \
	    eval.checkpoint_path=kuleshov-group/mdlm-owt \
	      data=openwebtext-split  \
	        model.length=32  \
		  sampling.predictor=ddpm  \
		    sampling.steps=1000 \
		      loader.eval_batch_size=1 \
		        sampling.num_sample_batches=1 \
			  backbone=hf_dit

