export CUDA_VISIBLE_DEVICES=0
python3 ./src/evaluate.py \
    --dataset cifar100 \
    --model resnet18 \
    --num_ensembles 10 \
    --device cuda \
    --eval_single \
    --batch_size 50 \
    --save_dir /home/sakai/projects/NDE/DistilEnsemble/wandb/run-20240617_180033-51x4eh3g/files