for type in Atest-Xtest A3-Xtest A4-Xtest 
do 
    echo $type
    for i in 16 8 4 2 1 0
    do 
        python seed.py logs/sbm-lam2.0-mu$i.0-n128-p16 $type 
    done
done
