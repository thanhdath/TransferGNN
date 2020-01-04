
# without transfer
python main.py --adj edgelist_path --labels label_path --features feature_path

# with transfer
python main.py --adj edgelist_path --labels label_path \
    --features feature_path  --transfer model_path
    