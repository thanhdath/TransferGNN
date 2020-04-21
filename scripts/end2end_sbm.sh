export CUDA_VISIBLE_DEVICES=1

for seed in $(seq 100 109)
do
    rm -r model/
    echo $seed
    datadir=data-transfers/synf-sbm-seed$seed
    mkdir logs
    LOGDIR=logs/end2end-sbm-seed$seed
    mkdir $LOGDIR
    echo "======================"
    echo "======================"
    echo "Training from scratch"
    for name in Atrain-Xtrain AtrainF-Xtrain 
    do 
        python -W ignore -u main.py --adj $datadir/$name/$name.txt --labels $datadir/$name/labels.txt \
            --features $datadir/$name/features.npz --seed $seed --epochs 300 > $LOGDIR/$name.log
    done

    echo "======================"
    echo "======================"
    echo "Transfer Atrain-Xtrain to others"
    for name in A2-Xtest A3-Xtest A4-Xtest Atest-Xtest
    do
        python -W ignore -u main.py --adj $datadir/$name/$name.txt --labels $datadir/$name/labels.txt \
            --features $datadir/$name/features.npz \
            --transfer model/Atrain-Xtrain.pkl --seed $seed --epochs 0 > $LOGDIR/$name-from-Atrain-Xtrain.log
    done

    echo "======================"
    echo "======================"
    echo "Transfer AtrainF-Xtrain to others"
    for name in A2-Xtest A3-Xtest A4-Xtest Atest-Xtest
    do
        python -W ignore -u main.py --adj $datadir/$name/$name.txt --labels $datadir/$name/labels.txt \
            --features $datadir/$name/features.npz \
            --transfer model/AtrainF-Xtrain.pkl --seed $seed --epochs 0 > $LOGDIR/$name-from-AtrainF-Xtrain.log
    done
done 


n=128
p=8
mkdir logs/end2end/
for lam in 0 0.5 1 1.5 1.75 2
do
    for mu in 0 1 2 4 8 16
    do
        python -u end2end.py --n $n --p $p --lam $lam --mu $mu > logs/end2end/sbm-n$n-p$p-lam$lam-mu$mu.log
    done
done 
