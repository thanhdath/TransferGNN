# GRAPH CLASSIFICATION + ADJ FUNCTION
mkdir logs/gc
for seed in $(seq 100 109)
do 
    mkdir logs/gc/synf-seed$seed
    for data in PROTEINS_full Synthie
    do 
        for f in ori knn sigmoid
        do 
            echo $data-$f
            python -u transfers/gc_fx.py --data $data --f $f --batch_size 128 --epochs 200 --seed $seed > logs/gc/synf-seed$seed/$data-$f.log
        done

    done
done 


# GRAPH CLASSIFICATION + INIT FEATURES
mkdir logs/gc
for seed in $(seq 100 104)
do
    mkdir logs/gc/seed$seed
    for data in DD DHFR ENZYMES FRANKENSTEIN IMDB-BINARY PROTEINS REDDIT-BINARY
    do
        for init in real svd degree degree-standard triangle kcore
        do
            echo $data-$init
            python -u transfers/gc_init_features.py --data $data --init $init --seed $seed > logs/gc/seed$seed/$data-$init.log
        done
    done 
done

# MNIST SUPERPIXELS & MNIST GRID
cd mnist
logdir=logs/mnist
mkdir $logdir
for seed in $(seq 100 104)
do
    python -u mnist.py --data grid --seed $seed > $logdir/grid-seed$seed.log 
    python -u mnist.py --data super --seed $seed > $logdir/super-seed$seed.log 
    python -u transfer.py --from_data grid --to_data super --seed $seed > $logdir/grid2super-seed$seed.log 
    python -u transfer.py --from_data super --to_data grid --seed $seed > $logdir/super2grid-seed$seed.log 
done


# TORUS & SPHERE
logdir=logs/torus-sphere
mkdir $logdir
for seed in $(seq 100 104)
do
    python transfers/torus_and_sphere.py --num-graphs 1000 \
        --from-data knn --to-data knn \
        --from-k 5 --to-k 5 --epochs 50 \
        --batch-size 128 --seed $seed > $logdir/from-knn-to-knn-seed$seed.log
    python -u transfers/torus_and_sphere.py --num-graphs 1000 \
        --from-data sigmoid --to-data sigmoid \
        --from-k 10 --to-k 10 --epochs 50 \
        --batch-size 128 --seed $seed > $logdir/from-sigmoid-to-sigmoid-seed$seed.log
    python -u transfers/torus_and_sphere.py --num-graphs 1000 \
        --from-data knn --to-data sigmoid \
        --from-k 5 --to-k 10  --epochs 50 \
        --batch-size 128 --seed $seed > $logdir/from-knn-to-sigmoid-seed$seed.log
    python -u transfers/torus_and_sphere.py --num-graphs 1000 \
        --from-data sigmoid --to-data knn \
        --from-k 10 --to-k 5  --epochs 50 \
        --batch-size 128 --seed $seed > $logdir/from-sigmoid-to-knn-seed$seed.log
done
