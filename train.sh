#!/bin/bash

make all

## Default settings
# ./learn_suc --itemlist data/itemlist.txt --behaviorlist data/behaviorlist.txt --output learn_suc-default.txt

## Complete settings
./learn_suc --itemlist data/itemlist.txt --behaviorlist data/behaviorlist.txt --output learn_suc-m1.txt --dim 128 --mode 1 --samples 1 --negative 10 --rate 0.025 --threads 8

make clean
