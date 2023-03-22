@echo off
SETLOCAL
CALL :StartPykeen1 ../KGs/FB15k-237 , Pykeen_DistMult
CALL :StartPykeen2 ../KGs/FB15k-237 , Pykeen_ComplEx
CALL :StartDice1 ../KGs/FB15k-237 , DistMult
CALL :StartDice2 ../KGs/FB15k-237 , ComplEx
EXIT /B %ERRORLEVEL%

@REM the hyperparameters are choosen according to https://github.com/pykeen/benchmarking


:StartPykeen1
python ../main.py --path_dataset_folder %~1 --model %~2 --num_epochs 100 --scoring_technique "NegSample" ^
--batch_size 256 --lr 0.001849662035249092 --embedding_dim 64 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"
:StartPykeen2
python ../main.py --path_dataset_folder %~1 --model %~2 --num_epochs 100 --scoring_technique "NegSample" ^
--batch_size 256 --lr 0.007525067744232913 --embedding_dim 256 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=256 loss="BCEWithLogitsLoss" ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"


:StartDice1
python ../main.py --path_dataset_folder %~1 --model %~2 --num_epochs 100 --scoring_technique "NegSample" ^
--batch_size 256 --lr 0.001849662035249092 --embedding_dim 64 --trainer "PL" --neg_ratio 1 ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"
:StartDice2
python ../main.py --path_dataset_folder %~1 --model %~2 --num_epochs 100 --scoring_technique "NegSample" ^
--batch_size 256 --lr 0.007525067744232913 --embedding_dim 256 --trainer "PL" --neg_ratio 1 ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"


EXIT /B 0