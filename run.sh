#!/bin/bash

#cp ../data/kftt/kftt_dev.txt ../data/kftt/train.txt
#cp ../data/kftt/kftt_devtest.txt ../data/kftt/dev.txt
#cp ../data/kftt/kftt_devtest.txt ../data/kftt/test.txt
#
#python wa2span_squad.py --do_lower --whole --wa_file ../data/kftt/train.txt --out ../data/kftt/train.json
#python wa2span_squad.py --do_lower --whole --wa_file ../data/kftt/dev.txt --out ../data/kftt/dev.json
#python wa2span_squad.py --do_lower --whole --wa_file ../data/kftt/test.txt --out ../data/kftt/test.json

# rm -r ~/.cache/huggingface/datasets/*
MODEL=bert-base-multilingual-cased
DATADIR='../data/'
SEED=42
TARGET='kftt'
OUTDIR='../out/'$TARGET/$SEED/
mkdir -p $OUTDIR
python my_run_qa.py --model_name_or_path $MODEL --data_dir $DATADIR$TARGET/ --version_2_with_negative --per_device_train_batch_size 12 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --max_answer_length 15 --do_predict --overwrite_cache --seed $SEED --output_dir $OUTDIR
for STAGE in dev test
  do
  python convert_start_end.py --questions $DATADIR$TARGET/$STAGE.json --nbest_predictions $OUTDIR$STAGE\_nbest_predictions.json --model_name_or_path $MODEL --out $OUTDIR$STAGE\_charindex_nbest_predictions.json
  python get_alignment.py --bidi_threshold --do_lower --alignments $DATADIR$TARGET/$STAGE.txt --nbest_predictions $OUTDIR$STAGE\_charindex_nbest_predictions.json --out $OUTDIR$STAGE\_bidi_th.out
  done