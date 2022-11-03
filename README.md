# Pytorch Implementation of QA Aligner
A pytorch implementation of [A Supervised Word Alignment Method based on Cross-Language Span Prediction using Multilingual BERT](https://aclanthology.org/2020.emnlp-main.41.pdf). 

Our implementation is based on the [original implementation](https://github.com/nttcslab-nlp/word_align) on TensorFlow by the authors and [HuggingFace Transformer examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering). 

## Notable changes
- While the original implementation truncates ```src (question)``` when the input sequence length exceeds the ```max_seq_len```, ours truncates ```tgt (context)```.

- Initial seed can be easily set with ```--seed``` argument. 

## How to use
Explain the usage with [KFTT (Kyoto Free Translation Task)](http://www.phontron.com/kftt/download/kftt-alignments.tar.gz) Japanese-English Alignment task as an example.

### Corpus preparation 
Please follow the instruction of the [original implementation](https://github.com/nttcslab-nlp/word_align): download the corpus into a ```data/kftt/``` directory and split it with ```make_kftt_data.sh```.

Then rename each split (naming matters in the later codes).
```
cp ../data/kftt/kftt_dev.txt ../data/kftt/train.txt
cp ../data/kftt/kftt_devtest.txt ../data/kftt/dev.txt
cp ../data/kftt/kftt_devtest.txt ../data/kftt/test.txt
```
Caution: We use ```devtest``` for both validation and testing, which is of course unacceptable in practice. This is for this demo only!

Generate SQuAD 2.0 style inputs:
```
python wa2span_squad.py --do_lower --whole --wa_file ../data/kftt/train.txt --out ../data/kftt/train.json
python wa2span_squad.py --do_lower --whole --wa_file ../data/kftt/dev.txt --out ../data/kftt/dev.json
python wa2span_squad.py --do_lower --whole --wa_file ../data/kftt/test.txt --out ../data/kftt/test.json
```

### Word alignment
The alignment process is conducted in 3 steps.
1. Train a QA model and predict word alignments with ```my_run_qa.py ```
2. Convert predicted alignment indices from ones of subwords to characters
3. Finally, convert character indices into word alignment 

```
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
```




