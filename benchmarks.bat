
echo run benchmarks...

python main.py --path_dataset_folder "KGs/UMLS" --model "Pykeen_DistMult" --num_epochs 10 --scoring_technique "NegSample" ^
--batch_size 1024 --lr 0.1 --embedding_dim 64 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=64 loss="bcewithlogits" ^
--use_SLCWALitModule --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "gpu" --devices 1
