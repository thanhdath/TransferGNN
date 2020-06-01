logdir=logs/transfer-ppi-multi/
mkdir $logdir
for seed in 100 101 102 103 104
do
    for f in ori knn sigmoid
    do
        python -u transfers/ppi_multi.py --seed $seed --f $f > $logdir/$f-seed$seed.log
    done
done
