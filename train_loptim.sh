TIME=$(date "+%Y-%m-%d-%H-%M-%S")

python train_loptim.py \
  --train_partition debug \
  --output_dir ./output/ft_velo/$TASK_NAME/$TIME/