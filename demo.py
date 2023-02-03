# Get a training dataset
from pykeen.datasets import UMLS, Nations
import torch
from pykeen import predict

dataset = Nations()
training_triples_factory = dataset.training

# Pick a model
from pykeen.models import DistMult

model = DistMult(
    triples_factory=training_triples_factory,
    loss="BCEWithLogitsLoss",
    embedding_dim=256,
)
x = torch.tensor([[ 0,  0,  0],
        [ 0,  0,  1],
        [ 0,  0,  2],
        [ 0,  0,  3],
        [ 0,  0,  4],
        [ 0,  0,  5],
        [ 0,  0,  6],
        [ 0,  0,  7],
        [ 0,  0,  8],
        [ 0,  0,  9],
        [ 0,  0, 10],
        [ 0,  0, 11],
        [ 0,  0, 12],
        [ 0,  0, 13]],dtype = torch.int32)

# Pick an optimizer from Torch
from torch.optim import Adam

optimizer = Adam(params=model.get_grad_params(), lr=0.01)

# Pick a training approach (sLCWA or LCWA)
from pykeen.training import SLCWATrainingLoop

training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training_triples_factory,
    optimizer=optimizer,
)

# Train like Cristiano Ronaldo
_ = training_loop.train(
    triples_factory=training_triples_factory,
    num_epochs=200,
    batch_size=1024,
)
# predictions_tails = predict.predict_target(model=model,head = x[0,0].item(),relation = x[0,1].item(),)
predictions_heads = predict.predict_target(model=model, relation = x[0,1].item(),tail=x[0,2].item(),targets=x[:,0])
print(predictions_heads)
# Pick an evaluator
from pykeen.evaluation import RankBasedEvaluator

evaluator = RankBasedEvaluator()

# Get triples to test
mapped_triples = dataset.testing.mapped_triples

# Evaluate
results = evaluator.evaluate(
    model=model,
    mapped_triples=mapped_triples,
    batch_size=1024,
    additional_filter_triples=[
        dataset.training.mapped_triples,
        dataset.validation.mapped_triples,
    ],
)
# print(results)
# print(results.get_metric("hits_at_10"))
# print(results.to_flat_dict())
