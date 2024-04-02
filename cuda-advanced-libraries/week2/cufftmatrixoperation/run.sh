#!/usr/bin/env bash
make clean build

make run ARGS="-n=1024" > output.txt