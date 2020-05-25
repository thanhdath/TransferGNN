mkdir logs
mkdir logs/transfer-real 
for seed in 100 101 102 103 104 
do
    mkdir logs/transfer-real/seed$seed/
    echo $seed 
    python -u test.py --seed $seed > logs/transfer-real/seed$seed/log.log
done
