# Code for numerical experiments
## For details, see [Pu, Timms, and Tam, _ArXiv preprint_, 2026](https://arxiv.org/abs/)

* algorithms.py - contains the PG-EXTRA algorithm [Shi, Ling, Wu, and Yin (2015)], the NIDS algorithm [Li, Shi, and Yan (2019)], and Algorithms 1 and 2 in Pu, Timms, and Tam (2026)
* lasso.ipynb - generates results for the group LASSO (l1-regularised least-squares) problem in Section 4.1
  * requires algorithms.py
* l1_logistic.ipynb - generates results for the distributed l1-regularised logistic regression problem in Section 4.2
  * requires algorithms.py
* dist_localisation.ipynb - generates results for the distributed sensor localisation problem in Section 4.3
  * requires algorithms.py and rls_synthetic (contains dataset)
