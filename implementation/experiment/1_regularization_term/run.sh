current_time=$(date "+%Y.%m.%d-%H.%M.%S")
result_filename=result_$current_time.csv
echo 'n_classes,classes,loss_function,learning_rate,lambda,final_loss,accuracy' >> $result_filename

for n_classes in 2 3 4 6 8 10 12
do
    echo '#################################### n_classes: '$n_classes >> log_$current_time.txt
    for i in {1..20}
    do
        python run.py --n_clusters $n_classes --loss 'reconstruction_loss_without_regularization' --lr 1e-4 --lamda 0 --filename $result_filename >> log_$current_time.txt
    done
    echo '--------------------------- with regularization term' >> log_$current_time.txt
    for regularization_term in 0.1 0.5 1 1.5 2
    do
        for i in {1..20}
        do
            python run.py --n_clusters $n_classes --loss 'reconstruction_loss_with_regularization' --lr 1e-4 --lamda $regularization_term --filename $result_filename >> log_$current_time.txt
        done
    done
done
