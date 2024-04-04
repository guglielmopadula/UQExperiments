A repo for testing uncertainty quantification methods using Bayesian Neural Networks, via numpyro.
|Model         |Time       |Mean Rel Interval Length        |Mean Error|
|--------------|-----------|--------------------------------|----------|
|LaplaceNormal |120        |0.10                            |0.002     |
|LaplaceUniform|119        |0.10                            |0.002     |
|GPR           |54         |0.09                            |0.001     |
|PCM           |14         |0.07                            |0.009     |
|SplitConformal|1.8        |0.09                            |0.006     |