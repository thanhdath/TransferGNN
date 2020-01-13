for seed in $(seq 100 109)
do
    # gen graphs
    python transfers/syn_reproducef.py --p 8 --mu 2000 --lam 1.1 --kind sigmoid \
        --threshold 0.7 --n 2560 \
        --n_graphs 3 --epochs 0 --seed $seed
    mkdir logs
    mkdir logs/synf-seed$seed
    datadir=data-transfers/synf
    for name in Atrain-Xtrain Atest-Xtest
    do
        python -W ignore main.py --adj $datadir/$name/$name.txt --labels $datadir/$name/labels.txt \
            --features $datadir/$name/features.npz --seed $seed > logs/synf-seed$seed/$name.log
    done
    for name in Atest-Xtest
    do
        python -W ignore main.py --adj $datadir/$name/$name.txt --labels $datadir/$name/labels.txt \
            --features $datadir/$name/features.npz \
            --transfer model/Atrain-Xtrain.pkl --seed $seed > logs/synf-seed$seed/$name-from.log
    done
done

Atest-Xtest  A2-Xtest A3-Xtest A4-Xtest 
same u, high n, low p, high mu => transfer feature-only ok p=8, n=2560, mu=2000, lam=1.1
