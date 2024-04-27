current_time=$(date "+%Y.%m.%d-%H.%M.%S")
result_filename=result_$current_time.csv
echo 'n_classes,classes,loss_function,learning_rate,lambda,final_loss,accuracy' >> $result_filename

for n_classes in 2 3 4 6 8 10 12
do
    echo '#################################### n_classes: '$n_classes >> log_$current_time.txt

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

        echo '################## using class: '$class_list >> log_$current_time.txt

        # 对每个生成的class列表 把所有不同的setting都跑一遍
        # without regularization term
        python run.py --n_clusters $n_classes --classes $class_list --loss 'reconstruction_loss_without_regularization' --lr 1e-4 --tol 1e-10 --lamda 0 --filename $result_filename >> log_$current_time.txt
        echo '--------------------------- with regularization term' >> log_$current_time.txt
        # with regularization term, for different lambda
        for regularization_term in 0.1 0.5 1 1.5 2
        do
            python run.py --n_clusters $n_classes --classes $class_list --loss 'reconstruction_loss_with_regularization' --lr 1e-4 --tol 1e-10 --lamda $regularization_term --filename $result_filename >> log_$current_time.txt
        done
    done
done
