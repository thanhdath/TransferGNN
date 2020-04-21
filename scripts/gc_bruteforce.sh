
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
mkdir logs/gc/syn-seed100
mkdir logs/gc/syn-seed101
mkdir logs/gc/syn-seed102
mkdir logs/gc/syn-seed103
mkdir logs/gc/syn-seed104
for data in AIDS BZR BZR_MD COIL-DEL COLLAB COX2 DBLP_v1 DHFR DD ENZYMES FRANKENSTEIN IMDB-BINARY IMDB-MULTI Letter-high Mutagenicity MSRC_21 MUTAG NCI109 PROTEINS REDDIT-BINARY       
do
    for init in triangle
    do
        python -u gc.py --data $data --init $init --seed $seed > logs/gc/syn-seed$seed/$data-$init.log
    done
done 
for seed in $(seq 100 104)
do
    for data in AIDS BZR BZR_MD COIL-DEL COLLAB COX2 DBLP_v1 DHFR DD ENZYMES FRANKENSTEIN IMDB-BINARY IMDB-MULTI Letter-high Mutagenicity MSRC_21 MUTAG NCI109 PROTEINS REDDIT-BINARY       
    do
        for init in random
        do
            python -u gc.py --data $data --init $init --seed $seed > logs/gc/syn-seed$seed/$data-$init.log
        done
    done 
done

mkdir logs
for data in AIDS BZR BZR_MD COIL-DEL COLLAB COX2 DBLP_v1 DHFR DD ENZYMES FRANKENSTEIN IMDB-BINARY IMDB-MULTI Letter-high Mutagenicity MSRC_21 MUTAG NCI109 PROTEINS REDDIT-BINARY       
do
    mkdir logs/$data
    python -u gc_learnfeatures.py --data $data > logs/$data/learn.log
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


