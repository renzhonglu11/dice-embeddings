



from pykeen.pipeline import pipeline
from pykeen.sampling import BasicNegativeSampler



pipeline_result = pipeline(
    model='Distmult',
    model_kwargs=dict(
      embedding_dim=64,
      
    ),
    dataset='umls',
    training_loop='sLCWA',
    training_kwargs=dict(
      batch_size = 128,  
      
    ),
    negative_sampler=BasicNegativeSampler,
    negative_sampler_kwargs=dict(
      num_negs_per_pos = 32,
    ),
    optimizer = 'adam',
    optimizer_kwargs = dict(
      lr = 0.01  
    ),
    epochs = 100,
    evaluator='RankBasedEvaluator',
    result_tracker='wandb',
    result_tracker_kwargs=dict(
        project='pykeen_project',
    ),
)


pipeline_result.save_to_directory('pykeen_Distmult_umls')


pipeline_result = pipeline(
    model='ComplEx',
  model_kwargs=dict(
      embedding_dim=64,
      
    ),
    dataset='umls',
    training_loop='sLCWA',
    training_kwargs=dict(
      batch_size = 256,  
      
    ),
    negative_sampler=BasicNegativeSampler,
    negative_sampler_kwargs=dict(
      num_negs_per_pos = 32,
    ),
    optimizer = 'adam',
    optimizer_kwargs = dict(
      lr = 0.1  
    ),
    epochs = 100,
    evaluator='RankBasedEvaluator',
    result_tracker='wandb',
    result_tracker_kwargs=dict(
        project='pykeen_project',
    ),
)


pipeline_result.save_to_directory('pykeen_ComplEx_umls')





pipeline_result = pipeline(
    model='Distmult',
    model_kwargs=dict(
      embedding_dim=128,
      
    ),
    dataset='kinships',
    training_loop='sLCWA',
    training_kwargs=dict(
      batch_size = 256,  
      
    ),
    negative_sampler=BasicNegativeSampler,
    negative_sampler_kwargs=dict(
      num_negs_per_pos = 32,
    ),
    optimizer = 'adam',
    optimizer_kwargs = dict(
      lr = 0.01  
    ),
    epochs = 100,
    evaluator='RankBasedEvaluator',
    result_tracker='wandb',
    result_tracker_kwargs=dict(
        project='pykeen_project',
    ),
)


pipeline_result.save_to_directory('pykeen_Distmult_kinships')


pipeline_result = pipeline(
    model='ComplEx',
  model_kwargs=dict(
      embedding_dim=128,
      
    ),
    dataset='kinships',
    training_loop='sLCWA',
    training_kwargs=dict(
      batch_size = 128,  
      
    ),
    negative_sampler=BasicNegativeSampler,
    negative_sampler_kwargs=dict(
      num_negs_per_pos = 32,
    ),
    optimizer = 'adam',
    optimizer_kwargs = dict(
      lr = 0.1  
    ),
    epochs = 100,
    evaluator='RankBasedEvaluator',
    result_tracker='wandb',
    result_tracker_kwargs=dict(
        project='pykeen_project',
    ),
)


pipeline_result.save_to_directory('pykeen_ComplEx_kinships')

