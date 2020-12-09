# Hierarchical Representation Learning in Graph Neural Networks with Node Decimation Pooling

This is the official implementation of

> "Hierarchical Representation Learning in Graph Neural Networks with Node Decimation Pooling"  
> F. M. Bianchi, D. Grattarola, L. Livi, C. Alippi (2019)

This repo contains the necessary scripts and methods to run the graph classification experiments presented in the paper
using the proposed Node Decimation Pooling.

If you use any of this code for your own research, please cite the paper with:

```
@article{bianchi2018hierarchical,
  title={Hierarchical Representation Learning in Graph Neural Networks with Node Decimation Pooling},
  author={Bianchi, Filippo Maria and Grattarola, Daniele and Livi, Lorenzo and Alippi, Cesare},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2020}
}
```

## Setup

The code has the following dependencies:

- Tensorflow < 2.0.0
- Keras <= 2.2.5
- Spektral == 0.1.2
- Networkx == 2.4
- Numpy
- Scipy
- Scikit-learn

All are available through the Python Package Index and can be installed with `pip`.

Before running the main script, download and extract in `data/classification` any of the datsets
available [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets), e.g.:

```bash
$ wget https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/PROTEINS.zip -P data/classification
cd data/classification
unzip PROTEINS.zip
``` 

Make sure to update the config dictionary at the beginning of the main script to match the dataset (e.g., `'PROTEINS'`).

## Running

To run the experiment:

```sh
python GC_main.py
```

An output folder will be automatically created with a logfile and the trained model's weights. 
