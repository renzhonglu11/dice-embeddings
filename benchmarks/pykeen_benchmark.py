# # Get a training dataset
# from pykeen.datasets import Nations
# dataset = Nations()
# training_triples_factory = dataset.training

# # Pick a model
# from pykeen.models import TransE
# model = TransE(triples_factory=training_triples_factory)

# # Pick an optimizer from Torch
# from torch.optim import Adam
# optimizer = Adam(params=model.get_grad_params())

# # Pick a training approach (sLCWA or LCWA)
# from pykeen.training import SLCWATrainingLoop
# training_loop = SLCWATrainingLoop(
#     model=model,
#     triples_factory=training_triples_factory,
#     optimizer=optimizer,
# )

# # Train like Cristiano Ronaldo
# _ = training_loop.train(
#     triples_factory=training_triples_factory,
#     num_epochs=5,
#     batch_size=256,
# )

# # Pick an evaluator
# from pykeen.evaluation import RankBasedEvaluator
# evaluator = RankBasedEvaluator()

# # Get triples to test
# mapped_triples = dataset.testing.mapped_triples

# # Evaluate
# results = evaluator.evaluate(
#     model=model,
#     mapped_triples=mapped_triples,
#     batch_size=1024,
#     additional_filter_triples=[
#         dataset.training.mapped_triples,
#         dataset.validation.mapped_triples,
#     ],
# )
# # print(results)



from pykeen.pipeline import pipeline
from pykeen.sampling import BasicNegativeSampler



pipeline_result = pipeline(
    model='Distmult',
    dataset='umls',
    training_loop='sLCWA',
    negative_sampler=BasicNegativeSampler,
    optimizer = 'adam',
    epochs = 100,
    evaluator='RankBasedEvaluator',
    result_tracker='wandb',
    result_tracker_kwargs=dict(
        project='pykeen_project',
    ),
)


pipeline_result.save_to_directory('nations_transe')

