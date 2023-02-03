@echo off
SETLOCAL
@REM CALL :StartPykeen KGs/UMLS , Pykeen_DistMult
@REM CALL :StartPykeen KGs/UMLS , Pykeen_ComplEx
CALL :StartDice KGs/UMLS , DistMult
CALL :StartDice KGs/UMLS , ConEx
@REM CALL :StartPykeen KGs/KINSHIP , Pykeen_DistMult
@REM CALL :StartPykeen KGs/KINSHIP , Pykeen_ComplEx
@REM CALL :StartDice KGs/KINSHIP , DistMult
@REM CALL :StartDice KGs/KINSHIP , DistMult
EXIT /B %ERRORLEVEL%

:StartPykeen
python main.py --path_dataset_folder %~1 --model %~2 --num_epochs 250 --scoring_technique "NegSample" ^
--batch_size 1024 --lr 0.1 --embedding_dim 256 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" ^
--use_SLCWALitModule --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "cpu" --devices 1

:StartDice
python main.py --path_dataset_folder %~1 --model %~2 --num_epochs 250 --scoring_technique "NegSample" ^
--batch_size 1024 --lr 0.1 --embedding_dim 256 --trainer "PL" --neg_ratio 1 ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "cpu" --devices 1


EXIT /B 0


@REM echo run benchmarks...
@REM SET path1=KGs/UMLS
@REM SET path2=KGs/KINSHIP
@REM SET pykeenModel1=Pykeen_DistMult
@REM SET pykeenModel2=Pykeen_ComplEx
@REM SET diceModel1=DistMult
@REM SET diceModel2=DistMult

@REM python main.py --path_dataset_folder %path1% --model %pykeenModel1% --num_epochs 250 --scoring_technique "NegSample" ^
@REM --batch_size 1024 --lr 0.1 --embedding_dim 256 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=64 loss="CrossEntropy" ^
@REM --use_SLCWALitModule --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "gpu" --devices 1

@REM python main.py --path_dataset_folder %path1% --model %pykeenModel2% --num_epochs 250 --scoring_technique "NegSample" ^
@REM --batch_size 1024 --lr 0.1 --embedding_dim 256 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=64 loss="CrossEntropy" ^
@REM --use_SLCWALitModule --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "gpu" --devices 1

@REM python main.py --path_dataset_folder %path1% --model %diceModel1% --num_epochs 250 --scoring_technique "NegSample" ^
@REM --batch_size 1024 --lr 0.1 --embedding_dim 256 --trainer "torchDDP" --neg_ratio 1 ^
@REM --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 

@REM python main.py --path_dataset_folder %path1% --model %diceModel2% --num_epochs 250 --scoring_technique "NegSample" ^
@REM --batch_size 1024 --lr 0.1 --embedding_dim 256 --trainer "torchDDP" --neg_ratio 1 ^
@REM --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 