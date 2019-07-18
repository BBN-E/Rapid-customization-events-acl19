#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH/dist && /anaconda-py3/envs/customized_event/bin/python -m http.server 5051
