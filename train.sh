#!/bin/bash

make all

## Default settings
# ./learn_suc --itemlist data/itemlist.txt --behaviorlist data/behaviorlist.txt --output learn_suc-default.emb.txt

## Complete settings
./learn_suc --itemlist data/itemlist.txt --behaviorlist data/behaviorlist.txt --output learn_suc-m1-s100.emb.txt --size 128 --mode 1 --samples 100 --negative 10 --rho 0.025 --threads 8

make clean
