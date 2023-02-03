
from pykeen.hpo import hpo_pipeline
from pykeen.datasets.base import PathDataset


path ='../KGs/Nations'
train_path = path + "/train.txt"
test_path = path + "/test.txt"
valid_path = path + "/valid.txt"

dataset = PathDataset(
            training_path=train_path,
            testing_path=test_path,
            validation_path=valid_path,
            create_inverse_triples=True,  # NodePiece need in inverse triple
        )


def func():
    hpo_pipeline_result = hpo_pipeline(
        n_trials=10,
        dataset=dataset,
        model='DistMult',
        training_loop='sLCWA',
        lr_scheduler='ExponentialLR',
        optimizer='SGD',
        evaluator_kwargs=dict(filtered=True),
        lr_scheduler_kwargs_ranges=dict(
        gamma=dict(type=float, low=0.3, high=1.0),
    ),
        model_kwargs_ranges=dict(
        embedding_dim=dict(type=int, low=16, high=256, step=32)
    ),
    )
    return 

if __name__ == '__main__':
    func()
  
