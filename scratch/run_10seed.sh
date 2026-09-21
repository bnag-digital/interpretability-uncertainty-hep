#!/bin/bash
# Ten-seed extension: train and explain seeds 5-9 on both datasets, draw the
# figures into figures_10seed/, and print the tables. No stage passes --force
# or --floor-repeats, so the five-seed runs already on disk are skipped rather
# than rewritten. Launch it detached:
#   setsid nohup bash scratch/run_10seed.sh > /tmp/RUN10.log 2>&1 < /dev/null &
set -u
cd /work/users/bnag/interpretability-uncertainty-hep || exit 1

PY=./.pixi/envs/default/bin/python
SEEDS="5 6 7 8 9"
STAGES=logs/run10_stages.log
mkdir -p logs/run10 logs/tables_10seed

# one log per stage, and a single file recording where the job has got to
stage () {
    name=$1
    shift
    echo "BEGIN $name $(date -u +%FT%TZ)" >> $STAGES
    "$@" > logs/run10/$name.log 2>&1
    rc=$?
    echo "END   $name rc=$rc $(date -u +%FT%TZ)" >> $STAGES
    if [ $rc -ne 0 ]; then
        echo "FAILED at $name, stopping" >> $STAGES
        exit $rc
    fi
}

echo "START $(date -u +%FT%TZ)" >> $STAGES

# training. The flags reproduce what seeds 0-4 were trained with: the jets sit
# on the particle view's split at patience 10, covertype on depth 8 and the
# wider DNN at patience 20.
stage train_jets $PY scripts/train.py --models bdt dnn gnn pfn efn \
    --seeds $SEEDS --n-jets 200000 --jet-source h5 --patience 10
stage train_covertype $PY scripts/train.py --models bdt dnn --dataset covertype \
    --n-jets 200000 --seeds $SEEDS --max-depth 8 --hidden 256 128 64 --lr 0.001 \
    --patience 20

# explaining. Every explainer each model has, ceiling only.
stage explain_jets $PY scripts/explain.py --models bdt dnn gnn pfn efn \
    --seeds $SEEDS --train-level n200000 --quiet
stage explain_covertype $PY scripts/explain.py --models bdt dnn --dataset covertype \
    --seeds $SEEDS --train-level covertype_n200000 --quiet

# figures, into a new directory so figures/ keeps the five-seed set
stage figures_jets $PY scripts/make_figures.py --figure-dir figures_10seed
stage figures_covertype $PY scripts/make_figures.py --dataset covertype \
    --figure-dir figures_10seed
stage figures_covertype_cls1 $PY scripts/make_figures.py --dataset covertype \
    --cls-idx 1 --figure-dir figures_10seed

# tables, written next to the five-seed copies under logs/
for ds in hls4ml covertype; do
    stage tables_paper_$ds $PY scratch/paper_tables.py --dataset $ds
    stage tables_levels_$ds $PY scratch/levels_table.py --dataset $ds
    stage tables_model_$ds $PY scratch/model_table.py --dataset $ds
    cp logs/run10/tables_paper_$ds.log logs/tables_10seed/paper_$ds.txt
    cp logs/run10/tables_levels_$ds.log logs/tables_10seed/levels_$ds.txt
    cp logs/run10/tables_model_$ds.log logs/tables_10seed/model_$ds.txt
done
stage tables_paper_covertype_cls1 $PY scratch/paper_tables.py --dataset covertype --cls-idx 1
cp logs/run10/tables_paper_covertype_cls1.log logs/tables_10seed/paper_covertype_cls1.txt

echo "ALL DONE $(date -u +%FT%TZ)" >> $STAGES
