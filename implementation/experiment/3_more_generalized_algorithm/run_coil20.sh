dataset=coil20
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
result_filename=${dataset}_${current_time}.csv
echo 'dataset,n_classes,classes,tol,acc,nmi' >> $result_filename

for n_classes in 2 10 20
do
    echo '#################################### n_classes: '$n_classes >> ${dataset}_$current_time.txt

    if [ $n_classes -eq 20 ]; then
        repeat=1
    else
        repeat=10
    fi
    for i in $(seq 1 $repeat) # 每个setting重复10次
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
        CUDA_VISIBLE_DEVICES=0 python -u run.py --dataset $dataset --n_clusters $n_classes --classes $class_list --filename $result_filename >> ${dataset}_$current_time.txt
    done
done
