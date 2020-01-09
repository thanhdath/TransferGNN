mu=$1
lam=$2
n=256
p=128
for seed in $(seq 100 109)
do
    # gen graphs
    python transfers/synthetic.py --mu $mu --lam $lam --p $p --n1 $n --n2 $n --seed $seed
    python transfers/syn_learnD.py --seed $seed --data data-transfers/syn-seed$seed/

    # X-related , transfer XA 
    log=logs/transfers-synXA-mu$mu-lam$lam-n$n-p$p
    mkdir $log
    # feature only X
    for i in 0 1 
    do 
        data=data-transfers/syn-seed$seed/$i
        python -W ignore -u main.py --adj $data/syn$i.txt --labels $data/labels.txt \
            --features $data/features.npz --epochs 500 --feature-only > $log/seed$seed-$i-feature.log
    done
    # transfer
    for i in 0 1 
    do 
        data=data-transfers/syn-seed$seed/$i
        python -W ignore -u main.py --adj $data/syn$i.txt --labels $data/labels.txt \
            --features $data/features.npz --epochs 500 > $log/seed$seed-$i.log
    done 
    data=data-transfers/syn-seed$seed/0
    python -W ignore -u main.py --adj $data/syn0.txt --labels $data/labels.txt \
            --features $data/features.npz --transfer model/syn1.pkl > $log/seed$seed-0-from-1.log
    data=data-transfers/syn-seed$seed/1
    python -W ignore -u main.py --adj $data/syn1.txt --labels $data/labels.txt \
            --features $data/features.npz --transfer model/syn0.pkl > $log/seed$seed-1-from-0.log

    # D-related, transfer DA
    log=logs/transfers-synDA-mu$mu-lam$lam-n$n-p$p
    mkdir $log
    # feature only D
    for i in 0 1 
    do 
        data=data-transfers/synD-seed$seed/$i
        python -W ignore -u main.py --adj $data/$i.txt --labels $data/labels.txt \
            --features $data/features.npz --epochs 500 --feature-only > $log/seed$seed-$i-feature.log
    done
    # transfer
    for i in 0 1 
    do 
        data=data-transfers/synD-seed$seed/$i
        python -W ignore -u main.py --adj $data/$i.txt --labels $data/labels.txt \
            --features $data/features.npz --epochs 500 > $log/seed$seed-$i.log
    done 
    data=data-transfers/synD-seed$seed/0
    python -W ignore -u main.py --adj $data/0.txt --labels $data/labels.txt \
            --features $data/features.npz --transfer model/1.pkl --epochs 500 > $log/seed$seed-0-from-1.log
    data=data-transfers/synD-seed$seed/1
    python -W ignore -u main.py --adj $data/1.txt --labels $data/labels.txt \
            --features $data/features.npz --transfer model/0.pkl --epochs 500 > $log/seed$seed-1-from-0.log
done
