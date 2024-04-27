dataset=reuters
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
result_filename=${dataset}_${current_time}.csv
echo 'dataset,n_classes,classes,tol,acc,nmi' >> $result_filename

for n_classes in 2 8 15
do
    echo '#################################### n_classes: '$n_classes >> ${dataset}_$current_time.txt

    for i in {1..10} # 每个setting重复10次
    do
        # generate n_classes random numbers between 1 and 20.
        numbers=($(shuf -i 1-41 -n $n_classes))  # 生成随机不重复数字列表
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
        python -u run.py --dataset $dataset --n_clusters $n_classes --classes $class_list --filename $result_filename >> ${dataset}_$current_time.txt
    done
done
