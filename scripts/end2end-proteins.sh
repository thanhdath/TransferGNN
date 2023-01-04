# GEN PKL FILE
for seed in 100 101 102 103 104
do
    echo $seed 
    python process_data/process_proteins.py --seed $seed 
done

# END2END
for gnn in mean gcn sgc
do 
    logdir=logs/end2end-$gnn
    mkdir $logdir
    for seed in 100 101 102 103 104
    do
        mkdir logs/end2end
        python process_data/process_proteins.py --seed $seed
        python -u end2end.py --data-path data/proteins_full-seed$seed.pkl \
            --ckpt-dir ckpt-proteins/gat-$gnn/seed$seed --epochs 500 --hidden 64 --classify-weight 1 --adj-weight 1 \
            --embedding_dim 32 --seed $seed --batch-size 64 --gnn1 gat --gnn2 $gnn --n-layers 3 > $logdir/proteins-seed$seed.log
    done
done 

# NORMAL TRAINING
for gnn in mean gcn sgc
do 
    logdir=logs/normal-$gnn
    mkdir $logdir
    for seed in 100 101 102 103 104
    do
        python -u normal_training.py --data-path data/proteins_full-seed$seed.pkl \
            --epochs 500 --hidden 64 --seed $seed --batch-size 64 --gnn $gnn > $logdir/proteins-seed$seed.log
    done
done
