
# GEN PKL FILE
python process_data/process_ppi.py

# END2END
for gnn in mean gcn sgc
do 
    logdir=logs/end2end-$gnn
    mkdir $logdir
    for seed in 100 101 102 103 104
    do 
        python -u end2end.py --data-path data/ppi.pkl \
            --ckpt-dir ckpt-ppi/gat-$gnn/seed$seed --epochs 1000 \
            --hidden 128 --classify-weight 1 --adj-weight 1 \
            --embedding_dim 128 --seed $seed --batch-size 4 \
            --gnn1 gat --gnn2 $gnn --n-layers 3 > $logdir/ppi-seed$seed.log
    done 
done

# NORMAL TRAINING
for gnn in gcn sgc mean
do 
    logdir=logs/normal-$gnn
    mkdir $logdir
    for seed in 100 101 102 103 104
    do
        python -u normal_training.py --data-path data/ppi.pkl \
            --epochs 500 --hidden 128 --seed $seed --batch-size 4 --gnn $gnn > $logdir/ppi-seed$seed.log
    done
done
