# Code for numerical experiments
## For details, see [Pu, Timms, and Tam, _ArXiv preprint_, 2026](https://arxiv.org/abs/)
_Generating figures_
* algorithms.py - contains the PG-EXTRA algorithm [Shi, Ling, Wu, and Yin (2015)], the NIDS algorithm [Li, Shi, and Yan (2019)], and Algorithms 1 and 2 in Pu, Timms, and Tam (2026)
* psse.ipynb - generates results for one instance of the distributed power system estimation problem in Section 4.1
  * requires algorithms.py and dict_of_psse_instances
* l1logistic.ipynb - generates results for one instance of the distributed l1-regularised logistic regression problem in Section 4.2
  * requires algorithms.py and dataset from http://www.csie.ntu.edu.tw/~cjlin/libsvm

_Generating table of results_
* algorithms_comparison.py - contains the PG-EXTRA algorithm [Shi, Ling, Wu, and Yin (2015)], the NIDS algorithm [Li, Shi, and Yan (2019)], and Algorithms 1 and 2 in Pu, Timms, and Tam (2026)
* psse_comparison.ipynb - generates table of results for the distributed power system estimation problem in Section 4.1
  * requires algorithms_comparison.py and dict_of_psse_instances
* l1logistic_comparison.ipynb - generates table of results for the distributed l1-regularised logistic regression problem in Section 4.2
  * requires algorithms_comparison.py and dataset from http://www.csie.ntu.edu.tw/~cjlin/libsvm
