export DATASET_NAME=glue-mrpc
TIME=$(date "+%Y-%m-%d-%H-%M-%S")
CUDA_VISIBLE_DEVICES=0 python pt_crossfit.py \
    --dataset_name=$DATASET_NAME \
    --dataset_seed=13 \
    --output_dir ./output/pt_crossfit/$DATASET_NAME/$TIME/