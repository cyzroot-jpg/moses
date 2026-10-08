#!/bin/bash

set -e

# ==============================
# 配置
# ==============================

BASE_DIR="/data01/tuyilist/cyz"
PROJECT_DIR="/data01/tuyilist/cyz/moses/moses"

LOG_DIR="${BASE_DIR}/genoutput/logs"
DONE_FILE="${BASE_DIR}/genoutput/all_completed.txt"

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/run_all_$(date '+%Y%m%d_%H%M%S').log"

# ==============================
# 开始记录日志
# ==============================

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "任务开始: $(date)"
echo "日志文件: ${LOG_FILE}"
echo "=========================================="

cd "${PROJECT_DIR}"

# 如果之前存在完成文件，先删除
rm -f "${DONE_FILE}"


# ==============================
# 1. ORGAN sample
# ==============================

echo ""
echo "=========================================="
echo "[1/5] 开始执行 ORGAN sample"
echo "时间: $(date)"
echo "=========================================="

python scripts/sample.py organ \
       --model_load /data01/tuyilist/cyz/checkpoint/ORGAN/model.pt \
       --vocab_load /data01/tuyilist/cyz/checkpoint/ORGAN/vocab.pt \
       --config_load /data01/tuyilist/cyz/checkpoint/ORGAN/config.pt \
       --n_samples 30000 \
       --gen_save /data01/tuyilist/cyz/genoutput/organ/gen_tiem.csv \
       --time_path /data01/tuyilist/cyz/genoutput/time/smaple_organ.json

echo "[1/5] ORGAN sample 完成: $(date)"


# ==============================
# 2. VAE sample
# ==============================

echo ""
echo "=========================================="
echo "[2/5] 开始执行 VAE sample"
echo "时间: $(date)"
echo "=========================================="

python scripts/sample.py vae \
       --model_load /data01/tuyilist/cyz/checkpoint/VAE/model.pt \
       --vocab_load /data01/tuyilist/cyz/checkpoint/VAE/vocab.pt \
       --config_load /data01/tuyilist/cyz/checkpoint/VAE/config.pt \
       --n_samples 30000 \
       --gen_save /data01/tuyilist/cyz/genoutput/vae/gen_tiem.csv \
       --time_path /data01/tuyilist/cyz/genoutput/time/smaple_vae.json

echo "[2/5] VAE sample 完成: $(date)"


# ==============================
# 3. Char RNN eval
# ==============================

echo ""
echo "=========================================="
echo "[3/5] 开始执行 Char RNN eval"
echo "时间: $(date)"
echo "=========================================="

python scripts/eval.py \
       --test_path /data01/tuyilist/cyz/moses/moses/dataset/data/test.csv.gz \
       --test_scaffolds_path /data01/tuyilist/cyz/moses/moses/dataset/data/test_scaffolds.csv.gz \
       --train_path /data01/tuyilist/cyz/moses/moses/dataset/data/train.csv.gz \
       --gen_path /data01/tuyilist/cyz/genoutput/char_rnn/gen_tiem.csv \
       --device sdaa:0 \
       --n_jobs 8 \
       --time_path /data01/tuyilist/cyz/moses/time/evla_char_rnn.json

echo "[3/5] Char RNN eval 完成: $(date)"


# ==============================
# 4. ORGAN eval
# ==============================

echo ""
echo "=========================================="
echo "[4/5] 开始执行 ORGAN eval"
echo "时间: $(date)"
echo "=========================================="

python scripts/eval.py \
       --test_path /data01/tuyilist/cyz/moses/moses/dataset/data/test.csv.gz \
       --test_scaffolds_path /data01/tuyilist/cyz/moses/moses/dataset/data/test_scaffolds.csv.gz \
       --train_path /data01/tuyilist/cyz/moses/moses/dataset/data/train.csv.gz \
       --gen_path /data01/tuyilist/cyz/genoutput/organ/gen_tiem.csv \
       --device sdaa:0 \
       --n_jobs 8 \
       --time_path /data01/tuyilist/cyz/moses/time/evla_orgn.json

echo "[4/5] ORGAN eval 完成: $(date)"


# ==============================
# 5. VAE eval
# ==============================

echo ""
echo "=========================================="
echo "[5/5] 开始执行 VAE eval"
echo "时间: $(date)"
echo "=========================================="

python scripts/eval.py \
       --test_path /data01/tuyilist/cyz/moses/moses/dataset/data/test.csv.gz \
       --test_scaffolds_path /data01/tuyilist/cyz/moses/moses/dataset/data/test_scaffolds.csv.gz \
       --train_path /data01/tuyilist/cyz/moses/moses/dataset/data/train.csv.gz \
       --gen_path /data01/tuyilist/cyz/genoutput/vae/gen_tiem.csv \
       --device sdaa:0 \
       --n_jobs 8 \
       --time_path /data01/tuyilist/cyz/moses/time/evla_vae.json

echo "[5/5] VAE eval 完成: $(date)"


# ==============================
# 全部完成
# ==============================

echo ""
echo "=========================================="
echo "全部任务执行完成！"
echo "完成时间: $(date)"
echo "=========================================="

cat > "${DONE_FILE}" << EOF
ALL_TASKS_COMPLETED

completed_time=$(date '+%Y-%m-%d %H:%M:%S')

status=success

tasks=5
EOF

echo "完成标记已写入:"
echo "${DONE_FILE}"
echo ""
echo "任务全部结束: $(date)"
