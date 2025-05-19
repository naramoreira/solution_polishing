# Solution Polish

This repository contains a generic Evolutionary Algorithm to help solving hard MIP or CP models. It is based on a known idea of repeatedly do a local search for the optimal solution, iteratively changing and increasing the area of search.

It can be applied both to MIP and CP model. It requires a feasible initial solution to work, but in case you don't have one available, the polish algorithm can also try to find one before it starts the local search iterations.

A solver is required, and so far the supported options are Gurobi and Google CP-SAT.

The `toy_example.py` in the [examples](/examples/) folder shows how to plug and use the polish algorithm in an application.

The implementation of this algorithm is inspired by the following article:

Edward Rothberg, (2007) An Evolutionary Algorithm for Polishing Mixed Integer Programming Solutions. INFORMS Journal on Computing 19(4):534-541. https://doi.org/10.1287/ijoc.1060.0189