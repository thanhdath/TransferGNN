seed=100
python transfers/synthetic.py --mu 0.2 --p 128 --seed $seed
python transfers/syn_learnD.py --seed $seed --data data-transfers/syn-seed$seed/

mkdir logs/transfers-synXA
for i in 0 1 
do 
    data=data-transfers/syn-seed$seed/$i
    python -W ignore -u main.py --adj $data/syn$i.txt --labels $data/labels.txt \
        --features $data/features.npz --epochs 500 > logs/transfers-synXA/$i.log
done 
data=data-transfers/syn-seed$seed/0
python -W ignore -u main.py --adj $data/syn0.txt --labels $data/labels.txt \
        --features $data/features.npz --transfer model/syn1.pkl > logs/transfers-synXA/0-from-1.log
data=data-transfers/syn-seed$seed/1
python -W ignore -u main.py --adj $data/syn1.txt --labels $data/labels.txt \
        --features $data/features.npz --transfer model/syn0.pkl > logs/transfers-synXA/1-from-0.log


mkdir logs/transfers-synDA
for i in 0 1 
do 
    data=data-transfers/synD-seed$seed/$i
    python -W ignore -u main.py --adj $data/$i.txt --labels $data/labels.txt \
        --features $data/features.npz --epochs 500 > logs/transfers-synDA/$i.log
done 
data=data-transfers/synD-seed$seed/0
python -W ignore -u main.py --adj $data/0.txt --labels $data/labels.txt \
        --features $data/features.npz --transfer model/1.pkl --epochs 500 > logs/transfers-synDA/0-from-1.log
data=data-transfers/synD-seed$seed/1
python -W ignore -u main.py --adj $data/1.txt --labels $data/labels.txt \
        --features $data/features.npz --transfer model/0.pkl --epochs 500 > logs/transfers-synDA/1-from-0.log

