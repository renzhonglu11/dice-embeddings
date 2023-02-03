from pykeen.pipeline import pipeline
result = pipeline(dataset="nations", model="pairre", training_kwargs=dict(num_epochs=0))


from pykeen.datasets import get_dataset
from pykeen.predict import predict_triples
dataset = get_dataset(dataset="nations")
pack = predict_triples(model=result.model, triples=dataset.validation)

df = pack.process(factory=result.training).df
print(df)