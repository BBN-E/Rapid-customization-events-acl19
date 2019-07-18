#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH/../ && LD_LIBRARY_PATH=/anaconda-py3/lib:$LD_LIBRARY_PATH /anaconda-py3/envs/customized_event/bin/uwsgi --ini uwsgi.ini