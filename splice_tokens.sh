#!/bin/sh

STYLE=$1 bsub < splice_tokens.lsf -J $1 -o $1.log
