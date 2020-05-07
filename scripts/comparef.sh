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
