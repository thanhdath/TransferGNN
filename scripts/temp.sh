
for gnn in sgc gcn
do
    logdir=logs/sbm-$gnn
    mkdir $logdir
    for seed in 100 101 102 103 104
    do 
        mkdir $logdir/seed$seed
        python -u transfers/sbm_gc.py --lam 1 --mu 8 --p 32 \
            --n 128 --n-graphs 300 --epochs 300 --hidden 32 \
            --gnn $gnn --f ori > $logdir/seed$seed/log.log
    done
done  
