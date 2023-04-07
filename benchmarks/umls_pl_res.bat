

@REM :StartPykeen1
python ../main.py --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 256 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --use_SLCWALitModule --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam" --normalization "None"
@REM :StartPykeen2
python ../main.py --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --use_SLCWALitModule --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam" --normalization "None"


@REM :StartDice1
python ../main.py --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 256 --trainer "PL" --neg_ratio 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam" --normalization "None"
@REM :StartDice2
python ../main.py --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --trainer "PL" --neg_ratio 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam" --normalization "None"


