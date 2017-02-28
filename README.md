# MIQP Prototype Solver based on OSQP
The solver lives in the `miosqp.py` file. In order to run it you need to install the [OSQP Python interface](https://github.com/bstellato/osqp).

## Run example
In order to run the examples from to compare with CPLEX or GUROBI, after installing their respective python insterface, you need to install [mathprogbasepy](https://github.com/bstellato/mathprogbasepy). The random examples can be run from `example_miqp.py` file.

## Maximum number of iterations
For some problem instances, OSQP reaches the maximum number of iterations. In order to deal with them, they are dumped to different files in the `max_iter_examples` folder. In order to load them separately and solve them with OSQP, you can run the `run_maxiter_problem.py` file.


## TODO

-   [ ] Check when it hits the max number of iterations (if only when the subproblems are infeasible) -> Yes, that's the case
-   [ ] Find first solution using ADMM heuristic
-   [ ] Test binary-only problems -> Works really well!
-   [x] Add proper infeasibility test
-   [x] Add number of integer infeasible variables to printing function
-   [x] Cleanup unused functions/variables
-   [x] Write comments and descriptions of objects properly
-   [x] Add functions: add_left and add_right for each node
    -   [x] N.B. Update node l_i and u_i by restricting newly introduced ones in branching
-   [x] Store solution when upper bound improves (only x, dual does not matter)
-   [x] Add function to deal separately with root node (when branching)
-   [x] Complete get_bounds function
-   [x] Add proper warm starting (from parent node, when branching)
-   [x] Add only workspace passed to the nodes!
