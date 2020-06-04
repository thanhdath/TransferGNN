
# GEN SBM GRAPH
mkdir data-sbm
for seed in 100 101 102 103 104
do
    echo $seed
    python process_data/process_sbm.py --n 128 --p 8 --lam 1.5 --mu 8 --seed $seed
done

# END2END
for gnn in gcn sgc mean
do 
    mkdir data-sbm
    logdir=logs/end2end-$gnn
    mkdir $logdir
    for seed in 100 101 102 103 104
    do
        python process_data/process_sbm.py --n 128 --p 32 --lam 1 --mu 8 --seed $seed --n-graphs 300
        python -u end2end.py --data-path data-sbm/n128-p32-lam1.0-mu8.0-seed$seed.pkl \
            --ckpt-dir ckpt-sbm/gat/n128-p32-lam1.0-mu8.0-seed$seed/ --epochs 300 \
            --hidden 32 --classify-weight 1 --adj-weight 1 \
            --embedding_dim 32 --seed $seed --batch-size 64 --gnn1 gat --gnn2 $gnn --n-layers 3 > $logdir/sbm-seed$seed.log
    done
done

# NORMAL TRAINING
for gnn in gcn sgc mean
do 
    logdir=logs/normal-$gnn
    mkdir $logdir
    for seed in 100 101 102 103 104
    do
        python -u normal_training.py --data-path data-sbm/n128-p32-lam1.0-mu8.0-seed$seed.pkl \
            --epochs 300 --hidden 32 --seed $seed --batch-size 64 --gnn $gnn > $logdir/sbm-seed$seed.log
    done
done
