#!/bin/bash

#!/bin/bash

t1=(1 2 51 50 50)
t2=(2 3 68 32 34)
t3=(3 5 61 20 20)
t4=(4 6 68 15 17)
t5=(5 8 65 9 13)
t6=(6 10 66 1 11)
t7=(7 12 63 1 9)
t8=(8 13 64 4 8)
t9=(9 15 63 2 7)
t10=(10 17 60 4 6)

arrays=(t1 t2 t3 t4 t5 t6 t7 t8 t9 t10)


param=("${t1[@]}")
dataset=("MNIST" "FashionMNIST" "CIFAR10")
lr=(0.01 0.01 0.001)

echo $param
echo "${param[1]}"
