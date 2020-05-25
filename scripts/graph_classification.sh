
mkdir logs
for data in AIDS BZR BZR_MD COIL-DEL COLLAB COX2 DBLP_v1 DHFR DD ENZYMES FRANKENSTEIN IMDB-BINARY IMDB-MULTI Letter-high Mutagenicity MSRC_21 MUTAG NCI109 PROTEINS REDDIT-BINARY       
do
    mkdir logs/$data
    for init in one degree-standard degree-onehot
    do
        python -u gc.py --data $data --init $init > logs/$data/$init.log
    done
done 

mkdir logs/gc
for seed in $(seq 102 102)
do
    mkdir logs/gc/seed$seed
    for data in BZR BZR_MD COIL-DEL COLLAB COX2 DBLP_v1 DHFR DD ENZYMES FRANKENSTEIN IMDB-BINARY IMDB-MULTI Letter-high Mutagenicity MSRC_21 MUTAG NCI109 PROTEINS REDDIT-BINARY       
    do
        for init in svd degree kcore triangle deepwalk one
        do
            python -u transfers/gc_identity.py --data $data --init $init --seed $seed > logs/gc/seed$seed/$data-$init.log
        done
    done 
done

mkdir logs/gc
for seed in $(seq 100 109)
do 
    mkdir logs/gc/synf-seed$seed
    for data in PROTEINS_full Synthie
    do 
        for f in ori knn sigmoid
        do 
            echo $data-$f
            python -u transfers/graph_kernel.py --data $data --f $f --batch_size 128 --epochs 200 --seed $seed > logs/gc/synf-seed$seed/$data-$f.log
        done

    done
done 


mkdir logs/ppi-withoutf
for seed in $(seq 100 109)
do 
    mkdir logs/ppi-withoutf/synf-seed$seed
    f=random
    python -u ppi_withoutf.py --f $f --seed $seed > logs/ppi-withoutf/synf-seed$seed/$f.log
done 

# GRAPH CLASSIFICATION INIT FEATURES
mkdir logs/gc
for seed in $(seq 100 104)
do
    mkdir logs/gc/seed$seed
    for data in DD DHFR ENZYMES FRANKENSTEIN IMDB-BINARY PROTEINS REDDIT-BINARY
    do
        for init in real svd degree degree-standard triangle kcore
        do
            echo $data-$init
            python -u transfers/gc_identity.py --data $data --init $init --seed $seed > logs/gc/seed$seed/$data-$init.log
        done
    done 
done

# GRAPH CLASSIFICATION LEARN FEATURES + RECONSTRUCTION LOSS
mkdir logs
mkdir logs/gc-learnfeatures
for seed in $(seq 100 104)
do
    mkdir logs/gc-learnfeatures/seed$seed
    for data in DD DHFR ENZYMES FRANKENSTEIN IMDB-BINARY PROTEINS    
    do
        echo $data
        python -u transfers/gc_learnfeatures.py --name $data --seed $seed > logs/gc-learnfeatures/seed$seed/$data.log
    done 
done

REDDIT-BINARY
tmux a -t 1


mkdir logs/gc
for seed in $(seq 100 104)
do
    mkdir logs/gc/seed$seed
    for data in ENZYMES
    do
        for init in real
        do
            echo $data-$init
            python -u transfers/gc_identity.py --data $data --init $init --seed $seed > logs/gc/seed$seed/$data-$init.log
        done
    done 
done
