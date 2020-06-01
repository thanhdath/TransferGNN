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


cd transfergnn
export PYTHONPATH=.
conda activate transfer
seed=$1
for n in 128
do
    for p in 8 
    do
        for lam in 1.75 2.0
        do
            for mu in 0.0 1.0 2.0 4.0 8.0 16.0
            do
                mkdir logs/sbm-n$n-p$p-lam$lam-mu$mu/
                mkdir logs/sbm-n$n-p$p-lam$lam-mu$mu/seed$seed/
                for f in ori knn sigmoid random
                do
                    python -u transfers/sbm_gc.py --seed $seed --n $n --p $p --lam $lam --mu $mu --f $f > logs/sbm-n$n-p$p-lam$lam-mu$mu/seed$seed/$f.log
                done
            done
        done 
    done
done

for n in 32 64 128 256 512 1024
do
    for p in 8 
    do
        for lam in 1.0 1.5 2.0
        do
            for mu in 0.0 1.0 2.0 4.0 8.0 16.0
            do
                mkdir logs/sbm-n$n-p$p-lam$lam-mu$mu/
                mkdir logs/sbm-n$n-p$p-lam$lam-mu$mu/seed$seed/
                for f in ori knn sigmoid random
                do
                    python -u transfers/sbm_gc.py --seed $seed --n $n --p $p --lam $lam --mu $mu --f $f > logs/sbm-n$n-p$p-lam$lam-mu$mu/seed$seed/$f.log
                done
            done
        done 
    done
done

for n in 128
do
    for p in 8 16 32 64 128 256
    do
        for lam in 1.0 1.5 2.0
        do
            for mu in 0.0 1.0 2.0 4.0 8.0 16.0
            do
                mkdir logs/sbm-n$n-p$p-lam$lam-mu$mu/
                mkdir logs/sbm-n$n-p$p-lam$lam-mu$mu/seed$seed/
                for f in ori knn sigmoid random
                do
                    python -u transfers/sbm_gc.py --seed $seed --n $n --p $p --lam $lam --mu $mu --f $f > logs/sbm-n$n-p$p-lam$lam-mu$mu/seed$seed/$f.log
                done
            done
        done 
    done
done


n=128
p=8
for lam in 0.0 0.5 1.0 1.5 1.75 2.0
do
    for mu in 0.0 1.0 2.0 4.0 8.0 16.0
        do
        basedir=/home/datht/transfergnn/data/model-block/gen-sbm-n$n-p$p-lam$lam-mu$mu/
        mkdir logs/sbm-modelblock-n$n-p$p-lam$lam-mu$mu
        for seed in $(seq 100 100)
        do 
            echo $seed
            rm -r model/
            datadir=$basedir/synf-seed100-multigraphs
            LOGDIR=logs/sbm-modelblock-n$n-p$p-lam$lam-mu$mu/synf-seed100-multigraphs
            mkdir $LOGDIR
            echo "======================"
            echo "======================"
            echo "Training from scratch"
            for name in Atrain-Xtrain AtrainF-Xtrain
            do 
                python -W ignore -u sbm.py --input-dir $datadir/$name --seed $seed --epochs 200 > $LOGDIR/$name.log
            done

            echo "======================"
            echo "======================"
            echo "Transfer Atrain-Xtrain to others"
            for name in Atest-Xtest A2-Xtest A3-Xtest X4-Xtest
            do
                python -W ignore -u sbm.py --is-test-graphs --input-dir $datadir/$name \
                    --transfer model/Atrain-Xtrain.pkl --seed $seed --epochs 0 > $LOGDIR/$name-from-Atrain-Xtrain.log
            done

            echo "======================"
            echo "======================"
            echo "Transfer AtrainF-Xtrain to others"
            for name in A2-Xtest A3-Xtest A4-Xtest Atest-Xtest
            do
                python -W ignore -u sbm.py --is-test-graphs --input-dir $datadir/$name \
                    --transfer model/AtrainF-Xtrain.pkl --seed $seed --epochs 0 > $LOGDIR/$name-from-AtrainF-Xtrain.log
            done
        done
    done
done




export CUDA_VISIBLE_DEVICES=0,1

for lam in 0.0 0.5 1.0 1.5 1.75 2.0
do 
    for mu in 0.0 1.0 2.0 4.0 8.0 16.0
    do
        basedir=/home/datht/gran/sbm/lam$lam-mu$mu/
        mkdir logs
        # mu2-lam0 mu2-lam0.5 mu2-lam1 mu2-lam1.5 mu2-lam2
        mkdir logs/sbm-lam$lam-mu$mu
        for seed in $(seq 100 100)
        do
            echo $seed
            rm -r model/
            datadir=$basedir/synf-seed100-multigraphs
            LOGDIR=logs/sbm-lam$lam-mu$mu/synf-seed100-multigraphs
            mkdir $LOGDIR
            echo "======================"
            echo "======================"
            echo "Training from scratch"
            for name in Atrain-Xtrain
            do 
                python -W ignore -u ppi.py --input-dir $datadir/$name --seed $seed --epochs 300 > $LOGDIR/$name.log
            done

            echo "======================"
            echo "======================"
            echo "Transfer Atrain-Xtrain to others"
            for name in Atest-Xtest A3-Xtest A4-Xtest 
            do
                python -W ignore -u ppi.py --is-test-graphs --input-dir $datadir/$name \
                    --transfer model/Atrain-Xtrain.pkl --seed $seed --epochs 0 > $LOGDIR/$name-from-Atrain-Xtrain.log
            done
        done
    done 
done


lam=2.0
for n in 32 64 128 256
do
    for mu in 0.0 1.0 2.0 4.0 8.0 16.0
    do
        basedir=/home/datht/gran/sbm/lam$lam-mu$mu-n$n/
        mkdir logs
        # mu2-lam0 mu2-lam0.5 mu2-lam1 mu2-lam1.5 mu2-lam2
        mkdir logs/sbm-lam$lam-mu$mu-n$n
        for seed in $(seq 100 100)
        do
            echo $seed
            rm -r model/
            datadir=$basedir/synf-seed100-multigraphs
            LOGDIR=logs/sbm-lam$lam-mu$mu-n$n/synf-seed100-multigraphs
            mkdir $LOGDIR
            echo "======================"
            echo "======================"
            echo "Training from scratch"
            for name in Atrain-Xtrain
            do 
                python -W ignore -u ppi.py --input-dir $datadir/$name --seed $seed --epochs 300 > $LOGDIR/$name.log
            done

            echo "======================"
            echo "======================"
            echo "Transfer Atrain-Xtrain to others"
            for name in Atest-Xtest A3-Xtest A4-Xtest 
            do
                python -W ignore -u ppi.py --is-test-graphs --input-dir $datadir/$name \
                    --transfer model/Atrain-Xtrain.pkl --seed $seed --epochs 0 > $LOGDIR/$name-from-Atrain-Xtrain.log
            done
        done
    done 
done


for lam in 1.0 1.5 2.0
do
    for n in 128
    do
        for mu in 0.0 1.0 2.0 4.0 8.0 16.0
        do
            for p in 8 16 32 64 128 256
            do
                basedir=/home/datht/gran/sbm/lam$lam-mu$mu-n$n-$p/
                mkdir logs
                mkdir logs/sbm-lam$lam-mu$mu-n$n-p$p
                for seed in $(seq 100 100)
                do
                    echo $seed
                    rm -r model/
                    datadir=$basedir/synf-seed100-multigraphs
                    LOGDIR=logs/sbm-lam$lam-mu$mu-n$n-p$p/synf-seed100-multigraphs
                    mkdir $LOGDIR
                    echo "======================"
                    echo "======================"
                    echo "Training from scratch"
                    for name in Atrain-Xtrain
                    do 
                        python -W ignore -u ppi.py --input-dir $datadir/$name --seed $seed --epochs 300 > $LOGDIR/$name.log
                    done

                    echo "======================"
                    echo "======================"
                    echo "Transfer Atrain-Xtrain to others"
                    for name in Atest-Xtest A3-Xtest A4-Xtest 
                    do
                        python -W ignore -u ppi.py --is-test-graphs --input-dir $datadir/$name \
                            --transfer model/Atrain-Xtrain.pkl --seed $seed --epochs 0 > $LOGDIR/$name-from-Atrain-Xtrain.log
                    done
                done
            done
        done 
    done
done

# for lam in 1.0 1.5 2.0
# do
#     for n in 128
#     do
#         for mu in 0.0 1.0 2.0 4.0 8.0 16.0
#         do
#             for p in 8 16 32 64 128 256
#             do
#                  python sbm.py --lam $lam --mu $mu --n $n --p $p
#             done
#         done 
#     done
# done


n=128
p=32
mu=8
lam=1
logdir=logs/sbm-n$128-p$p-mu$mu-lam$lam
mkdir $logdir
for seed in $(seq 100 104)
do
    python -u transfers/sbm_gc.py --seed $seed --n $n --p $p --lam $lam --mu $mu --f ori --gnn mlp > $logdir/seed$seed.log
done
