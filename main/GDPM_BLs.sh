#!/bin/sh

for graph in  TwitterSmall TwitterLarge; do
    julia main_GDPM_BLs.jl $graph 0.1 10 150
done
