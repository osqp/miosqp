# Mixed-Integer Quadratic Program Solver Based on OSQP

miOSQP solves an mixed-integer quadratic programs (MIQPs) of the form
```
minimize        0.5 x' P x + q' x

subject to      l <= A x <= u
                x[i] in Z for i in i_idx
                i_l[i] <= x[i] <= i_u[i] for i in i_idx
```
where `i_idx` is a vector of indices of which variables are integer and `i_l`, `i_u` are the lower and upper bounds on the integer variables respectively.


## Installation
To install the package simply run
```
python setup.py install
```
it depends on [OSQP](https://osqp.readthedocs.io), numpy and scipy.


## Usage
To solve a MIQP we need to run
```python
import miosqp
m = miosqp.MIOSQP()
m.setup(P, q, A, l, u, i_idx, i_l, i_u)
results = m.solve()
```
where `P` is a symmetric positive semidefinite matrix and `A` a matrix.
`P` and `A` are both in the scipy sparse CSC format.

The returned object `results` contains
-  `x`: the solution
-  `upper_glob`: the cost function upper bound
-  `run_time`: the solution time
-  `status`: the status
-  `osqp_solve_time`: the OSQP solve time as a percentage of the total solution time
-  `osqp_iter_avg`: the OSQP average number of iterations for each QP sub-problem solution


### Update problem vectors
Problem vectors can be updated without running the setup again. It can be done with
```python
m.update_vectors(q=q_new, l=l_new, u=u_new)
```

### Set initial solution guess
The initial guess can speedup the branch-and-bound algorithm significantly.
To set an initial feasible solution `x0` we can run
```python
m.set_x0(x0)
```

## Citing

If you are using this package for your work, please cite the [following paper](https://stellato.io/assets/downloads/publications/2018/miosqp_ecc.pdf):

```
@inproceedings{stellato2018,
  author = {Stellato, B. and Naik, V. V. and Bemporad, A. and Goulart, P. and Boyd, S.},
  title = {Embedded Mixed-Integer Quadratic Optimization Using the {OSQP} Solver},
  booktitle = {European Control Conference ({ECC})},
  year = {2018},
  code = {https://github.com/oxfordcontrol/miosqp},
  month = jul,
  groups = {power electronics, integer programs}
}
```

## Run examples
In order to run the examples from to compare with GUROBI, after installing the python insterface, you need to install [mathprogbasepy](https://github.com/bstellato/mathprogbasepy). Examples can be found in the `examples` folder.

-   Random MIQPs
-   Power system example

Note that you need [pandas](http://pandas.pydata.org/) package for storing the results dataframe and [tqdm](https://github.com/noamraph/tqdm) package for the progress bar.


<!-- ## Maximum number of iterations -->
<!-- For some problem instances, OSQP reaches the maximum number of iterations. In order to deal with them, they are dumped to different files in the `max_iter_examples` folder. In order to load them separately and solve them with OSQP, you can run `examples/run_maxiter_problem.py` file. -->
