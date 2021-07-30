<p align="center">
<img style="vertical-align:middle" src="https://raw.githubusercontent.com/Adapter-Hub/adapter-transformers/master/adapter_docs/logo.png" />
</p>
<h1 align="center">
<span>Transformer Adapter Project
</span>
</h1>

<h5 align="center">
The team will be working on the Transformer AdapterCapstone Project idea from Facebook.  The project aims to replicate the results of Gurungan et al. (2020)(https://arxiv.org/pdf/2004.10964.pdf), which uses domain and task adaptation prior to fine-tuning to increase classification  performance on a range of NLP tasks. We investigate the use of adapters and perform slightly better than the baseline results obtained by Gurungan et.al. for Task-Adaptive pretraining. We show that by using adapters, a parameter-efficient method for transfer learning with transformers, pre-trained models like RoBERTa base demonstrate an improvement in  performance  on  low-resource classification tasks.  Overall, adapters offer similar if not better performance at the cost of reduced training time and significantly less parameter-tuning.
</h5>

## Requirements

Transformer adapter Expermiment_ currently supports **Python 3.6+** and **PyTorch 1.7.0+**.
After [installing PyTorch](https://pytorch.org/get-started/locally/), you can install _adapter-transformers_ from PyPI ...

```
pip install -U adapter-transformers
```

## Getting Started

Project was implemented using Notebooks.

To get the augmented data for the CS domain from **kaggle arXiv dataset**, use this link for the [kaggle notebook](https://www.kaggle.com/sagjounkani/arxiv-metadata-exploration)
 

To get started with adapters, refer to these locations:

 **[TAPT (lang-adapter-training.ipynb)](https://github.gatech.edu/rirani6/cs7643_DRS/blob/master/lang-adapter-training.ipynb)**, a notebook which will download the task datasets and 
 train a language-adapter using MLM and then store it in file system. This adapter will be later used in our experiment for comparision of Task adaptation
 
 **[Task Adaptation Notebook (TaskAdapterNotebook.ipynb)](https://github.gatech.edu/rirani6/cs7643_DRS/blob/master/TaskAdapterNotebook.ipynb)**, This is our primary notebook a step by step execution for executing all our experiments and storing the results into a summary and detailed file. The confgiuration for the task adapter can be modified using the CustomConfig dataclass.

```
@dataclass
class CustomConfig(AdapterConfig):
    """
    The adapter architecture proposed by Pfeiffer et. al., 2020.
    Described in https://arxiv.org/pdf/2005.00247.pdf.
    """

    original_ln_before: bool = True
    original_ln_after: bool = True
    residual_before_ln: bool = True
    adapter_residual_before_ln: bool = False
    ln_before: bool = False
    ln_after: bool = False
    mh_adapter: bool = False
    output_adapter: bool = True
    non_linearity: str = "relu"
    reduction_factor: int = 16
    invertible_adapter: Optional[dict] = InvertibleAdapterConfig(
        block_type="nice", non_linearity="relu", reduction_factor=2
    )
```

## Authors:
 
 Sagar Jounkani, Dhruv Mehta, Ruzbeh Irani
