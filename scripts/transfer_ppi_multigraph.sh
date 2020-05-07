
basedir=/home/datht/gnf/gen-ppi3
mkdir logs
# mu2-lam0 mu2-lam0.5 mu2-lam1 mu2-lam1.5 mu2-lam2
mkdir logs/ppi-gat
for seed in $(seq 100 100)
do
    echo $seed
    rm -r model/
    datadir=$basedir/synf-seed100-multigraphs
    LOGDIR=logs/ppi-gat/synf-seed100-multigraphs
    mkdir $LOGDIR
    echo "======================"
    echo "======================"
    echo "Training from scratch"
    for name in Atrain-Xtrain AtrainF-Xtrain
    do 
        python -W ignore -u ppi.py --gat  --th 0.5 --input-dir $datadir/$name --seed $seed --epochs 100 > $LOGDIR/$name.log
    done

    echo "======================"
    echo "======================"
    echo "Transfer Atrain-Xtrain to others"
    for name in Atest-Xtest A2-Xtest
    do
        python -W ignore -u ppi.py  --gat  --th 0.5 --is-test-graphs --input-dir $datadir/$name \
            --transfer model/Atrain-Xtrain.pkl --seed $seed --epochs 0 > $LOGDIR/$name-from-Atrain-Xtrain.log
    done

    echo "======================"
    echo "======================"
    echo "Transfer AtrainF-Xtrain to others"
    for name in A2-Xtest Atest-Xtest
    do
        python -W ignore -u ppi.py  --gat  --th 0.5 --is-test-graphs --input-dir $datadir/$name \
            --transfer model/AtrainF-Xtrain.pkl --seed $seed --epochs 0 > $LOGDIR/$name-from-AtrainF-Xtrain.log
    done
done



basedir=/home/datht/gnf/gen-ppi2
mkdir logs
# mu2-lam0 mu2-lam0.5 mu2-lam1 mu2-lam1.5 mu2-lam2
mkdir logs/ppi-mean
for seed in $(seq 100 100)
do
    echo $seed
    rm -r model/
    datadir=$basedir/synf-seed100-multigraphs
    LOGDIR=logs/ppi-mean/synf-seed100-multigraphs
    mkdir $LOGDIR
    echo "======================"
    echo "======================"
    echo "Training from scratch"
    for name in Atrain-Xtrain AtrainF-Xtrain
    do 
        python -W ignore -u ppi_completed.py --input-dir $datadir/$name --seed $seed --epochs 100 > $LOGDIR/$name.log
    done

    echo "======================"
    echo "======================"
    echo "Transfer Atrain-Xtrain to others"
    for name in Atest-Xtest A2-Xtest
    do
        python -W ignore -u ppi_completed.py --is-test-graphs --input-dir $datadir/$name \
            --transfer model/Atrain-Xtrain.pkl --seed $seed --epochs 0 > $LOGDIR/$name-from-Atrain-Xtrain.log
    done

    echo "======================"
    echo "======================"
    echo "Transfer AtrainF-Xtrain to others"
    for name in A2-Xtest Atest-Xtest
    do
        python -W ignore -u ppi_completed.py --is-test-graphs --input-dir $datadir/$name \
            --transfer model/AtrainF-Xtrain.pkl --seed $seed --epochs 0 > $LOGDIR/$name-from-AtrainF-Xtrain.log
    done
done


export CUDA_VISIBLE_DEVICES=0,1
basedir=/home/datht/gran/exp/sbm128/GRANMixtureBernoulli_sbm/
mkdir logs
# mu2-lam0 mu2-lam0.5 mu2-lam1 mu2-lam1.5 mu2-lam2
mkdir logs/sbm
for seed in $(seq 100 100)
do
    echo $seed
    rm -r model/
    datadir=$basedir/synf-seed100-multigraphs
    LOGDIR=logs/sbm/synf-seed100-multigraphs
    mkdir $LOGDIR
    echo "======================"
    echo "======================"
    echo "Training from scratch"
    for name in Atrain-Xtrain AtrainF-Xtrain
    do 
        python -W ignore -u ppi.py --input-dir $datadir/$name --seed $seed --epochs 100 > $LOGDIR/$name.log
    done

    echo "======================"
    echo "======================"
    echo "Transfer Atrain-Xtrain to others"
    for name in Atest-Xtest A2-Xtest A3-Xtest A4-Xtest 
    do
        python -W ignore -u ppi.py --is-test-graphs --input-dir $datadir/$name \
            --transfer model/Atrain-Xtrain.pkl --seed $seed --epochs 0 > $LOGDIR/$name-from-Atrain-Xtrain.log
    done

    echo "======================"
    echo "======================"
    echo "Transfer AtrainF-Xtrain to others"
    for name in A2-Xtest A3-Xtest A4-Xtest Atest-Xtest
    do
        python -W ignore -u ppi.py --is-test-graphs --input-dir $datadir/$name \
            --transfer model/AtrainF-Xtrain.pkl --seed $seed --epochs 0 > $LOGDIR/$name-from-AtrainF-Xtrain.log
    done
done




