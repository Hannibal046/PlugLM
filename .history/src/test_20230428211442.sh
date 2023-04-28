# #!/bin/bash

# for str in "C语言中文网" "http://c.biancheng.net/" "成立7年了" "日IP数万"
# do
#     echo $str
# done


# for MODEL in "token" "sentence"
# do
# # echo $TASK 
# echo $MODEL
# done
for TASK in "citation_intent" "sciie"
do
    for MODEL in "token" "sentence"
    do
        echo $TASK 
        echo $MODEL
    done
done