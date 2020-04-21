

export CUDA_VISIBLE_DEVICES=0,1
basedir=/home/datht/gran/exp/sbm128/GRANMixtureBernoulli_sbm
mkdir logs
mkdir logs/ppi
for seed in $(seq 100 100)
do
    for i in $(seq 0 19)
    do
        echo $seed
        rm -r model/
        datadir=$basedir/synf-seed$seed-$i
        LOGDIR=logs/ppi/synf-seed$seed-$i
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
done
