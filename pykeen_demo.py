from pykeen.datasets import Nations
from pykeen.datasets.base import PathDataset,Dataset
from pykeen.triples.triples_factory import CoreTriplesFactory
# dataset = Nations()
# training_triples_factory = dataset.training

path = 'E:/DICE/dice-embeddings/KGs/Nations'
train_path = path + "/train.txt"
test_path = path + "/test.txt"
valid_path = path + "/valid.txt"
create_inverse_triples = False
PathDataset(
        training_path=train_path, testing_path=test_path, validation_path=valid_path,
        create_inverse_triples=create_inverse_triples,eager = True
)

CoreTriplesFactory()