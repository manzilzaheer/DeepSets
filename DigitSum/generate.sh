#!/bin/bash

mkdir -p data
cd data

if [ ! -f infimnist.tar.gz ]; then 
    wget http://leon.bottou.org/_media/projects/infimnist.tar.gz
else
    rm -rf infimnist
fi

tar xzvf infimnist.tar.gz
cd infimnist
make
./infimnist lab 10000 8109999 > ../mnist8m-labels-idx1-ubyte
./infimnist pat 10000 8109999 > ../mnist8m-patterns-idx3-ubyte
