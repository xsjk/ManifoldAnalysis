# /bin/bash

n_splits=5
n_jobs=120

max_n1=15
max_n2=15
max_k=30

for i in {0..9}
do
    for dist_type in default hausdorff sphere plane
    do
        for score_agg_method in mean min
        do
            uv run classifier.py --n_splits=$n_splits --random_state=$i --max_n1=$max_n1 --max_n2=$max_n2 --max_k=$max_k --dist_type=$dist_type --score_agg_method=$score_agg_method --output_dir=res/$dist_type/cluster2D/$score_agg_method --n_jobs=$n_jobs
        done
    done
done
