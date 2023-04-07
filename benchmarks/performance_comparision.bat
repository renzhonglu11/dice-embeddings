

python ../main.py --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.001 --embedding_dim 32 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --devices 1 --optim "Adam" --normalization "None"