dataset=coil20
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
result_filename=${dataset}_${current_time}.csv
echo 'dataset,n_classes,classes,tol,lambda,acc,nmi' >> $result_filename

for n_classes in 20 # 2 10 20
do
    echo '#################################### n_classes: '$n_classes >> ${dataset}_$current_time.txt

    for i in {1..10} # 每个setting重复10次
    do
        # generate n_classes random numbers between 1 and 20.
        numbers=($(shuf -i 1-20 -n $n_classes))  # 生成随机不重复数字列表
        class_list="["
        for ((i=0; i<${#numbers[@]}; i++))
        do
            class_list="${class_list}${numbers[i]}"
            if [ $i -lt $(( ${#numbers[@]} - 1 )) ]; then
                class_list="${class_list},"
            fi
        done
        class_list="${class_list}]"

        echo '################## using class: '$class_list >> ${dataset}_$current_time.txt

        # 对每个生成的class列表 把所有不同的setting都跑一遍
        for lambda in 20 # 1 5 10 15 20
        do
            python run.py --dataset $dataset --n_clusters $n_classes --classes $class_list --tol 1e-15 --lamda $lambda --filename $result_filename >> ${dataset}_$current_time.txt
        done
    done
done
