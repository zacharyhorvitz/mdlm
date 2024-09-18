python main.py \
  loader.batch_size=1 \
  loader.global_batch_size=1 \
  loader.num_workers=2 \
  loader.eval_batch_size=1 \
  model=small \
  data=openwebtext-split \
  wandb.name=mdlm-owt-sanity-2 \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=False \
  sampling.steps=1000
