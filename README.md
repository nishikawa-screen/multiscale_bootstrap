# pyscaleboot

This repository provides functions to perform multiscale bootstrap analysis for LiNGAM causal discovery results and estimate bootstrap probabilities for directed edges, bidirected arcs, and graphs.

When the repository root is on Python's import path (for example, when the notebook is in that directory), import the public API with:

```python
from pyscaleboot import MultiScaleBootstrap
```

If you use functions in this repository, install the Python packages `numpy`, `pandas`, and `lingam`. For AU/SI p-values, install R, set the path to `Rscript`, and install [`scaleboot`](https://github.com/shimo-lab/scaleboot) from its official GitHub repository.

## References

- Shimodaira, H. (2008). Testing regions with nonsmooth boundaries via multiscale bootstrap. *Journal of Statistical Planning and Inference*, 138, 1227-1241. [https://www.sciencedirect.com/science/article/abs/pii/S0378375807001504?via%3Dihub](https://www.sciencedirect.com/science/article/abs/pii/S0378375807001504?via%3Dihub)
- Komatsu, Y., Shimizu, S., and Shimodaira, H. (2010). Assessing Statistical Reliability of LiNGAM via Multiscale Bootstrap. *Artificial Neural Networks - ICANN 2010*, 309-314. [https://doi.org/10.1007/978-3-642-15825-4_40](https://doi.org/10.1007/978-3-642-15825-4_40)
