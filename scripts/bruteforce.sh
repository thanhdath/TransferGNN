mkdir logs/transfers

for seed in $(seq 100 109)
do
    mkdir logs/transfers-seed$seed/
    for i in $(seq 0 120)
    do
    echo $seed-$i
        mkdir logs/transfers-seed$seed/$i
        python transfers/ppi_learnD.py --seed $seed

        data=data-transfers/ppi/labels$i
        python -u -W ignore main.py --adj $data-0/labels$i-0.txt --labels $data-0/labels.txt \
            --features $data-0/features.npz > logs/transfers-seed$seed/$i/0.txt
        python -u -W ignore main.py --adj $data-1/labels$i-1.txt --labels $data-1/labels.txt \
            --features $data-1/features.npz > logs/transfers-seed$seed/$i/1.txt
        python -u -W ignore main.py --adj $data-1/labels$i-1.txt --labels $data-1/labels.txt \
            --features $data-1/features.npz --transfer model/labels$i-0.pkl > logs/transfers-seed$seed/$i/1-from-0.txt
        python -u -W ignore main.py --adj $data-0/labels$i-0.txt --labels $data-0/labels.txt \
            --features $data-0/features.npz --transfer model/labels$i-1.pkl > logs/transfers-seed$seed/$i/0-from-1.txt
    done
done
