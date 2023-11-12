#!/bin/bash

for i in {1..6}; do
	qsub fashionmnist_jobscript.sh
done

for i in {1..4}; do
	qsub mnist_jobscript.sh
done

for i in {1..10};do
	qsub cifar10_jobscript.sh
done
