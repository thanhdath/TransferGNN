data=n128-p8-lam1.5-mu8.0
mkdir logs
mkdir logs/sbm-gnf
mkdir logs/sbm-gnf/$data
path=/home/datht/gnf/ckpt-sbm/$data/gnf/sbm_gnf.pkl
for setting in A B C D 
do
    for model in sum
    do
        echo $setting-$model 
        python -u compare.py --data-path $path \
            --epochs 200 --setting $setting --th 0.5 --model $model > logs/sbm-gnf/$data/$model-setting-$setting.log
    done
done 



data=n128-p8-lam1.5-mu8.0
mkdir logs
mkdir logs/sbm-gran
mkdir logs/sbm-gran/$data
path=/home/datht/gran/exp/sbm128/GRANMixtureBernoulli_sbm/sbm_gran.pkl
for setting in A B C D 
do
    for model in mean sum gcn sgc gat
    do
        echo $setting-$model 
        python -u compare.py --data-path $path \
            --epochs 200 --setting $setting --th 0.5 --model $model > logs/sbm-gran/$data/$model-setting-$setting.log
    done
done 


data=n128-p8-lam1.5-mu8.0
mkdir logs
mkdir logs/sbm-graphrnn
mkdir logs/sbm-graphrnn/$data
path=/home/datht/graphrnn/graphs/sbm_graphrnn.pkl
for setting in A B C D 
do
    for model in mean sum gcn sgc gat
    do
        echo $setting-$model 
        python -u compare.py --data-path $path \
            --epochs 200 --setting $setting --th 0.5 --model $model > logs/sbm-graphrnn/$data/$model-setting-$setting.log
    done
done 


# PROTEINS_full

mkdir logs
mkdir logs/proteins-graphrnn
# path=/home/datht/gnf/ckpt-proteins/gnf/gnf_generated_graphs.pkl
path=/home/datht/graphrnn/graphs/sbm_graphrnn.pkl
for setting in A B C D 
do
    for model in mean sum gcn sgc gat
    do
        echo $setting-$model 
        python -u compare.py --data-path $path \
            --epochs 300 --setting $setting --th 0.5 --model $model --num_features 29 --num_labels 3  > logs/proteins-graphrnn/$model-setting-$setting.log
    done
done 

mkdir logs/proteins-gran
# path=/home/datht/gnf/ckpt-proteins/gnf/gnf_generated_graphs.pkl
path=/home/datht/gran/exp/proteins_full/GRANMixtureBernoulli_proteins_full/sbm_gran.pkl
for setting in A B C D 
do
    for model in mean sum gcn sgc gat
    do
        echo $setting-$model 
        python -u compare.py --data-path $path \
            --epochs 300 --setting $setting --th 0.5 --model $model --num_features 29 --num_labels 3  > logs/proteins-gran/$model-setting-$setting.log
    done
done 


# PARSE LOG FILE
for m in mean gat sum sgc gcn
do
    echo $m
    for s in A B C D
    do  
        tail -n1 $m-setting-$s.log
    done
done


# PPI 
mkdir logs
mkdir logs/ppi-gnf
# path=/home/datht/gnf/ckpt-proteins/gnf/gnf_generated_graphs.pkl
path=/home/datht/gnf/ckpt-ppi-old/gnf/gnf_generated_graphs.pkl
# for setting in A B C D 
# do
#     for model in mean sum gcn sgc
#     do
#         echo $setting-$model 
#         python -u compare.py --data-path $path \
#             --epochs 300 --setting $setting --th 0.5 --model $model --multiclass --num_features 50 --num_labels 121 --batch_size 4 --hidden 64  > logs/ppi-gnf/$model-setting-$setting.log
#     done
# done 

for setting in B C D A
do
    for model in gat
    do
        echo $setting-$model 
        python -u compare.py --data-path $path \
            --epochs 300 --setting $setting --th 0.5 --model $model --multiclass --num_features 50 --num_labels 121 --batch_size 1 --hidden 64 --nodes-bs 256  > logs/ppi-gnf/$model-setting-$setting.log
    done
done 

mkdir logs/ppi-gran
path=/home/datht/gran/exp/ppi/GRANMixtureBernoulli_ppi/sbm_gran.pkl
for setting in A B C D 
do
    for model in mean sum gcn sgc gat
    do
        echo $setting-$model 
        python -u compare.py --data-path $path \
            --epochs 300 --setting $setting --th 0.5 --model $model --multiclass --num_features 50 --num_labels 121 --batch_size 4 --hidden 64  > logs/ppi-gran/$model-setting-$setting.log
    done
done 

# TEMPORARILY
data=sbm
f=gnf
mkdir logs/$data-$f-deepwalk
for seed in 100
do
    mkdir logs/$data-$f-deepwalk/seed$seed
    path=/home/datht/gnf/ckpt-sbm/n128-p8-lam1.5-mu8.0-deepwalk-seed$seed/gnf/gnf_generated_graphs.pkl
    for setting in A B C D 
    do
        for model in mean sum gcn sgc gat
        do
            echo $setting-$model 
            python -u compare_modelf.py --data-path $path \
                --epochs 300 --setting $setting --th 0.5 --model $model \
                --num_features 40 --num_labels 2 --batch_size 32 \
                --hidden 64  > logs/$data-$f-deepwalk/seed$seed/$model-setting-$setting.log
        done
    done 
done 


# TEST TRANSFER WITH 4 SETTINGS - SBM
data=sbm
f=gnf
mkdir logs/$data-$f
for seed in 100 101 102 103 104
do
    mkdir logs/$data-$f/seed$seed
    path=/home/datht/gnf/ckpt-sbm/n256-p8-lam1.5-mu8.0-seed$seed/gnf/gnf_generated_graphs.pkl
    for setting in A B C D 
    do
        for model in mean sum gcn sgc gat
        do
            echo $setting-$model 
            python -u compare_modelf.py --data-path $path \
                --epochs 300 --setting $setting --th 0.5 --model $model \
                --num_features 8 --num_labels 2 --batch_size 32 \
                --hidden 64  > logs/$data-$f/seed$seed/$model-setting-$setting.log
        done
    done 
done 

data=sbm
f=graphrnn
mkdir logs/$data-$f
for seed in 100 101 102 103 104
do
    mkdir logs/$data-$f/seed$seed
    path=/home/datht/graphrnn/ckpt-sbm-seed$seed/graphs/sbm_graphrnn.pkl
    for setting in A B C D 
    do
        for model in mean sum gcn sgc gat
        do
            echo $setting-$model 
            python -u compare_modelf.py --data-path $path \
                --epochs 300 --setting $setting --th 0.5 --model $model \
                --num_features 8 --num_labels 2 --batch_size 32 \
                --hidden 64  > logs/$data-$f/seed$seed/$model-setting-$setting.log
        done
    done 
done 

data=sbm
f=gran
mkdir logs/$data-$f
for seed in 100 101 102 103 104
do
    mkdir logs/$data-$f/seed$seed
    path=/home/datht/gran/exp/sbm-seed100/GRANMixtureBernoulli_sbm/sbm_gran.pkl
    for setting in A B C D 
    do
        for model in mean sum gcn sgc gat
        do
            echo $setting-$model 
            python -u compare_modelf.py --data-path $path \
                --epochs 300 --setting $setting --th 0.5 --model $model \
                --num_features 8 --num_labels 2 --batch_size 32 \
                --hidden 64  > logs/$data-$f/seed$seed/$model-setting-$setting.log
        done
    done 
done 


# TEST TRANSFER WITH 4 SETTINGS - PROTEINS
data=proteins_full
f=gnf
mkdir logs/$data-$f
for seed in 101 102 103 104
do
    mkdir logs/$data-$f/seed$seed
    path=/home/datht/gnf/ckpt-proteins/seed$seed/gnf/gnf_generated_graphs.pkl
    for setting in A B C D 
    do
        for model in mean sum gcn sgc gat
        do
            echo $setting-$model 
            python -u compare_modelf.py --data-path $path \
                --epochs 300 --setting $setting --th 0.5 --model $model \
                --num_features 29 --num_labels 3 --batch_size 64 \
                --hidden 64 > logs/$data-$f/seed$seed/$model-setting-$setting.log
        done
    done 
done 

data=proteins_full
f=graphrnn
mkdir logs/$data-$f
for seed in 100 101 102 103 104
do
    mkdir logs/$data-$f/seed$seed
    path=/home/datht/graphrnn/ckpt-proteins-seed$seed/graphs/sbm_graphrnn.pkl
    for setting in A B C D 
    do
        for model in mean sum gcn sgc gat
        do
            echo $setting-$model 
            python -u compare_modelf.py --data-path $path \
                --epochs 300 --setting $setting --th 0.5 --model $model \
                --num_features 29 --num_labels 3 --batch_size 64 \
                --hidden 64 > logs/$data-$f/seed$seed/$model-setting-$setting.log
        done
    done 
done 

data=proteins_full
f=gran
mkdir logs/$data-$f
for seed in 100 101 102 103 104
do
    mkdir logs/$data-$f/seed$seed
    path=/home/datht/gran/exp/proteins_full-seed$seed/GRANMixtureBernoulli_proteins_full/sbm_gran.pkl
    for setting in A B C D 
    do
        for model in mean sum gcn sgc gat
        do
            echo $setting-$model 
            python -u compare_modelf.py --data-path $path \
                --epochs 300 --setting $setting --th 0.5 --model $model \
                --num_features 29 --num_labels 3 --batch_size 64 \
                --hidden 64 > logs/$data-$f/seed$seed/$model-setting-$setting.log
        done
    done 
done 

# TEST TRANSFER WITH 4 SETTINGS - PPI

data=ppi
f=gran
mkdir logs/$data-$f
for seed in 100 101 102 103 104
do
    mkdir logs/$data-$f/seed$seed
    path=/home/datht/gran/exp/ppi-seed$seed/GRANMixtureBernoulli_ppi/sbm_gran.pkl
    for setting in A B C D 
    do
        for model in gat
        do
            echo $setting-$model 
            python -u compare_modelf.py --multiclass --data-path $path \
                --epochs 300 --setting $setting --th 0.5 --model $model \
                --num_features 50 --num_labels 121 --batch_size 4 \
                --hidden 64 > logs/$data-$f/seed$seed/$model-setting-$setting.log
        done
    done 
done 

