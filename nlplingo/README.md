
#### Setting up nlplingo

NLPLingo uses Tensorflow as backend.

Setting up tensorflow CPU anaconda.
```
conda create --name tensorflow-1.5
source activate tensorflow-1.5
conda install -c anaconda python=2.7
conda install -c anaconda tensorflow=1.5
conda install keras=2.0.2
conda install -c anaconda future
conda install spacy
python -m spacy download en
```

Setting up tensorflow GPU anaconda.
```
conda create --name tensorflow-1.5-gpu
source activate tensorflow-1.5-gpu
conda install -c anaconda python=2.7
conda install -c anaconda tensorflow-gpu=1.5
conda install cudatoolkit=8.0
conda install keras=2.0.2
conda install -c anaconda future
conda install spacy
python -m spacy download en
```

Do ```export KERAS_BACKEND=tensorflow```


#### Running nlplingo

You need to point PYTHONPATH to 1 locations:
- dir where nlplingo resides

This is an example command:
PYTHONPATH=/home/repos/nlplingo python /home/repos/nlplingo/event/train_test.py --params x.params --mode train_trigger_from_file

Here are the available modes for train_test.py: 
- train_trigger_from_file
- test_trigger
- train_argument
- test_argument
- decode_trigger_argument


