


python -u encoder.py --key concat --epochs 300  --n-dup 1 --classify-weight 1.0 --adj-weight 1.0 --hidden 128 > logs/classify1-adj1.log
python -u encoder.py --key concat --epochs 300  --n-dup 1 --classify-weight 0.0 --adj-weight 1.0 > logs/classify0-adj1.log
python -u encoder.py --key concat --epochs 300  --n-dup 1 --classify-weight 1.0 --adj-weight 0.0 > logs/classify1-adj0.log

python -u encoder.py --key concat --epochs 300  --n-dup 1 --classify-weight 1.0 --adj-weight 1.0 --gat > logs/classify1-adj1-gat.log
python -u encoder.py --key concat --epochs 300  --n-dup 1 --classify-weight 0.0 --adj-weight 1.0 --gat > logs/classify0-adj1-gat.log
python -u encoder.py --key concat --epochs 300  --n-dup 1 --classify-weight 1.0 --adj-weight 0.0 --gat > logs/classify1-adj0-gat.log

# test encoder
python -u encoder.py --key concat --epochs 300  --n-dup 1 --classify-weight 1.0 --adj-weight 1.0 --hidden 128 --suffix 300 --test-gen 


python -u encoder2.py --key concat --epochs 300 --classify-weight 1.0 --adj-weight 1.0 --hidden 128
python -u encoder2.py --key concat --epochs 300 --classify-weight 1.0 --adj-weight 1.0 --hidden 128 --gat