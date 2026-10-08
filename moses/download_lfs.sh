#!/bin/bash
cd /data01/tuyilist/cyz/moses
base=https://media.githubusercontent.com/media/molecularsets/moses/master
for f in \
  moses/dataset/data/train.csv.gz \
  moses/dataset/data/test.csv.gz \
  moses/dataset/data/test_scaffolds.csv.gz \
  moses/dataset/data/test_stats.npz \
  moses/dataset/data/test_scaffolds_stats.npz ; do
  echo "downloading $f"
  curl -L "$base/$f" -o "$f"
done
