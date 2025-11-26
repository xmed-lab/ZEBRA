#!/bin/sh
export MASTER_PORT=$((RANDOM % 64512 + 1024))
export CUDA_VISIBLE_DEVICES=$1
export SSL_CERT_FILE=./cacert.pem
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
num_gpus=$((num_gpus + 1))
export PYTHONPATH=$(pwd)/:$PYTHONPATH


echo $num_gpus

EXP_ROOT_DIR="EXP"

exp=$2
stage=$3
subj=$4
ckpt=$5


if [ ! -d "./${EXP_ROOT_DIR}" ]; then
  mkdir ./${EXP_ROOT_DIR}
fi

if [ ! -d "./${EXP_ROOT_DIR}/exp_${exp}" ]; then
  mkdir ./${EXP_ROOT_DIR}/exp_${exp}
fi

if [ ! -d "./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj" ]; then
  mkdir ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj
fi

timestamp=$(date +"%H:%M_%Y-%m-%d")




if [[ "$stage" == *"1"* ]];
  then
    echo $stage
      accelerate launch --main_process_port $MASTER_PORT \
            train_$exp.py \
            --subj $subj \
            --batch_size 16 \
            --num_epochs 50 \
            --max_lr 1e-4 \
            --pretrain \
            --exp_dir ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj \
            | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/backbone_log_$timestamp.txt
fi



if [[ "$stage" == *"2"* ]];
  then
    echo $stage
      accelerate launch --main_process_port $MASTER_PORT \
            train_$exp.py \
            --subj $subj \
            --batch_size 16 \
            --num_epochs 50 \
            --max_lr 1e-4 \
            --exp_dir ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj \
            | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/prior_log_$timestamp.txt
fi


if [[ "$stage" == *"3"* ]];
  then
    python -u eval/recon.py --subj $subj --ckpt $ckpt --exp $exp --root_dir $EXP_ROOT_DIR | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/recon_log_$timestamp.txt
fi


if [[ "$stage" == *"4"* ]];
  then
    python -u eval/run_metrics.py --exp $exp --subj $subj --ckpt $ckpt --root_dir $EXP_ROOT_DIR | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/eval_res--$ckpt.txt
fi


if [[ "$stage" == *"5"* ]];
  then
    python -u eval/caption_keyframe.py --subj $subj --ckpt $ckpt --exp $exp --root_dir $EXP_ROOT_DIR | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/cap_log_$timestamp.txt

fi

if [[ "$stage" == *"6"* ]];
  then
    python -u eval/recon_enhance.py --subj $subj --exp $exp --ckpt $ckpt --root_dir $EXP_ROOT_DIR | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/recon_enhance_log_$timestamp.txt
fi

if [[ "$stage" == *"7"* ]];
  then
    python -u eval/run_metrics.py --mode enhance --exp $exp --subj $subj --ckpt $ckpt --root_dir $EXP_ROOT_DIR | tee ./${EXP_ROOT_DIR}/exp_${exp}/subj_$subj/eval_res_enhance--$ckpt.txt
fi




