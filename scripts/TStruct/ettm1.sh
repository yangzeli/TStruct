export OMP_NUM_THREADS=8

model_name=TStruct
dataset_name=ETTm1
log_name=ETTm1
log_dir="./logs/${model_name}_log"
mkdir -p "$log_dir"

configs=(
    "720  96   0.01  256  25  10  128  128  48  0.3  8  1.0  3"
    "720  192  0.01  128  25  10  64   64   72  0.3  8  1.0  1" 
    "720  336  0.01  128  25  10  32   32   72  0.4  8  1.0  1" 
    "720  720  0.01  64   25  10  128  128  48  0.3  8  0.8  3" 
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
    e_layers=${13} 
    model_id="${model_name}_${seq_len}_${pred_len}"

    echo "Running: model=${model_name}, seq_len=${seq_len}, pred_len=${pred_len}, lr=${lr}, bs=${bs}, \
    train_epochs=${train_epochs}, patience=${patience}, d_model=${d_model}, d_ff=${d_ff}, \
    patch_size=${patch_size}, dropout=${dropout}, rank=${rank}"

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ \
      --data_path ETTm1.csv \
      --model_id "$model_id" \
      --model "$model_name" \
      --data "$dataset_name" \
      --features M \
      --des 'Exp' \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
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
      --rank $rank \
      --e_layers $e_layers \
      --k_structure 5 \
      --attn_dropout 0.3 \
      --head_dropout 0.2 \
      --lradj type3 \
      --huber_lambda $huber_lambda \
      --auxi_lambda $auxi_lambda \
      > "${log_dir}/${model_name}_${log_name}_${seq_len}_${pred_len}.log" 2>&1
done
