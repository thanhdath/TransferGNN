n=128
p=8
lam=0
for mu in 0 1 2 4 8 16
do
    logdir=logs/sbm-n$n-p$p-mu$mu-lam$lam
    mkdir $logdir
    for seed in $(seq 100 104)
    do
        python -u transfers/sbm_gc.py --seed $seed --n $n --p $p --lam $lam --mu $mu --f ori --gnn mlp > $logdir/seed$seed.log
    done
done
