for seed in $(seq 100 119)
do
    # gen graphs
    echo $seed
    mkdir logs
    mkdir logs/synf-seed$seed
    python -u transfers/syn_reproducef.py --p 8 --mu 100 --lam 1.1 --kind sigmoid \
        --k 5 --n 16 --n_graphs 32 --epochs 1000 --seed $seed > logs/synf-seed$seed/syn.log

    datadir=data-transfers/synf-seed$seed
    for name in Atrain-Xtrain A2-Xtest
    do
        python -W ignore -u main.py --adj $datadir/$name/$name.txt --labels $datadir/$name/labels.txt \
            --features $datadir/$name/features.npz --seed $seed > logs/synf-seed$seed/$name.log
    done
    for name in A2-Xtest
    do
        python -W ignore -u main.py --adj $datadir/$name/$name.txt --labels $datadir/$name/labels.txt \
            --features $datadir/$name/features.npz \
            --transfer model/Atrain-Xtrain.pkl --seed $seed --epochs 0 #> logs/synf-seed$seed/$name-from.log
            #  >> logs/synf-Atest-Xtest.log
    done
done

Atest-Xtest A2-Xtest A3-Xtest A4-Xtest 