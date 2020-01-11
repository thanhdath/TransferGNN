# for seed in $(seq 100 109)
# do
    # gen graphs
python transfers/syn_reproducef.py --p 128 --mu 100 --lam 1.1 --kind sigmoid --threshold 0.55 --n 256
mkdir logs
mkdir logs/synf
datadir=data-transfers/synf
for name in Atrain-Xtrain Atest-Xtest
do
    python main.py --adj $datadir/$name/$name.txt --labels $datadir/$name/labels.txt \
        --features $datadir/$name/features.npz > logs/synf/$name.log
done
for name in Atest-Xtest
do
    python main.py --adj $datadir/$name/$name.txt --labels $datadir/$name/labels.txt \
        --features $datadir/$name/features.npz \
        --transfer model/Atrain-Xtrain.pkl > logs/synf/$name-from.log
done
# done
A2-Xtest A3-Xtest A4-Xtest 