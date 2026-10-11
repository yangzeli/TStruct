export OMP_NUM_THREADS=8

model_name=TStruct
dataset_name=custom_electricity
log_name=electricity
log_dir="./logs/${model_name}_log"
mkdir -p "$log_dir"

configs=(
    "720  96   0.002  64  25  10  512  512  48  0.1  32  0.9"
    "720  192  0.002  64  25  10  512  512  48  0.1  32  0.9"
    "720  336  0.002  64  25  10  512  512  48  0.1  32  0.9"
    "720  720  0.002  64  25  10  512  512  48  0.1  32  0.9"
)


for config in "${configs[@]}"; do
    set -- $config
    seq_len=$1
    pred_len=$2
    lr=$3
    bs=$4
    train_epochs=$5
    patience=$6
    d_model=$7
    d_ff=$8
    patch_size=$9
    dropout=${10}
    rank=${11}
    huber_lambda=${12}
    auxi_lambda=$(python -c "from decimal import Decimal; import sys; print(Decimal(sys.argv[2]) - Decimal(sys.argv[1]))" "$huber_lambda" 1.0)
    model_id="${model_name}_${seq_len}_${pred_len}"

    echo "Running: model=${model_name}, seq_len=${seq_len}, pred_len=${pred_len}, lr=${lr}, bs=${bs}, \
    train_epochs=${train_epochs}, patience=${patience}, d_model=${d_model}, d_ff=${d_ff}, \
    patch_size=${patch_size}, dropout=${dropout}, rank=${rank}"

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ \
      --data_path electricity.csv \
      --model_id "$model_id" \
      --model "$model_name" \
      --data "$dataset_name" \
      --features M \
      --des 'Exp' \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --itr 1 \
      --batch_size $bs  \
      --learning_rate $lr \
      --train_epochs $train_epochs \
      --patience $patience \
      --norm_type 'standard' \
      --d_model $d_model \
      --d_ff $d_ff \
      --patch_size $patch_size \
      --stride $patch_size \
      --dropout $dropout \
      --head_dropout 0.0 \
      --rank $rank \
      --e_layers 3 \
      --k_structure 4 \
      --attn_dropout 0.05 \
      --lradj type3 \
      --huber_lambda $huber_lambda \
      --auxi_lambda $auxi_lambda \
      > "${log_dir}/${model_name}_${log_name}_${seq_len}_${pred_len}.log" 2>&1
done
