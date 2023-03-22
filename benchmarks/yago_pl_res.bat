@echo off
SETLOCAL
CALL :StartPykeen1 ../KGs/YAGO3-10 , Pykeen_DistMult @REM lcwa
CALL :StartPykeen2 ../KGs/YAGO3-10 , Pykeen_ComplEx @REM lcwa
CALL :StartPykeen3 ../KGs/YAGO3-10 , Pykeen_DistMult @REM slcwa
CALL :StartPykeen4 ../KGs/YAGO3-10 , Pykeen_ComplEx @REM slcwa
CALL :StartDice1 ../KGs/YAGO3-10 , DistMult
CALL :StartDice2 ../KGs/YAGO3-10 , ComplEx
EXIT /B %ERRORLEVEL%

@REM the hyperparameters are choosen according to https://github.com/pykeen/benchmarking

:StartPykeen1
python ../main.py --path_dataset_folder %~1 --model %~2 --num_epochs 100 --scoring_technique "NegSample" ^
--batch_size 4096 --lr 0.00113355532419969 --embedding_dim 256 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=256 loss="BCEWithLogitsLoss" ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"
:StartPykeen2
python ../main.py --path_dataset_folder %~1 --model %~2 --num_epochs 100 --scoring_technique "NegSample" ^
--batch_size 8192 --lr 0.001723135381847608 --embedding_dim 256 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=256 loss="BCEWithLogitsLoss" ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"
:StartPykeen3
python ../main.py --path_dataset_folder %~1 --model %~2 --num_epochs 100 --scoring_technique "NegSample" ^
--batch_size 4096 --lr 0.00113355532419969 --embedding_dim 256 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=256 loss="BCEWithLogitsLoss" ^
--use_SLCWALitModule --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"
:StartPykeen4
python ../main.py --path_dataset_folder %~1 --model %~2 --num_epochs 100 --scoring_technique "NegSample" ^
--batch_size 8192 --lr 0.001723135381847608 --embedding_dim 256 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=256 loss="BCEWithLogitsLoss" ^
--use_SLCWALitModule --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"

:StartDice1
python ../main.py --path_dataset_folder %~1 --model %~2 --num_epochs 200 --scoring_technique "NegSample" ^
--batch_size 4096 --lr 0.00113355532419969 --embedding_dim 256 --trainer "PL" --neg_ratio 1 ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"
:StartDice2
python ../main.py --path_dataset_folder %~1 --model %~2 --num_epochs 200 --scoring_technique "NegSample" ^
--batch_size 8192 --lr 0.001723135381847608 --embedding_dim 256 --trainer "PL" --neg_ratio 1 ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"


EXIT /B 0