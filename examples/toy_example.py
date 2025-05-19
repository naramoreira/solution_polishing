"""This is a toy example of usage of the solution polish algorithm.

The purpose of this file is just to demonstrate how the polish can be called given
a MIP or CP model and an initial feasible solution.
"""

from typing import Any

import src.polish as pol
from src.optimiser import Optimiser, OptimiserSoftware


class MyModel(Optimiser):
    def __init__(self):
        super().__init__(optimiser=OptimiserSoftware.ORTOOLS_CPSAT)

    def build_model(self) -> tuple[pol.VarsSol, list[Any]]:
        """Builds a simple linear model, and returns a feasible solution and a list of
        variables that can be fixed by the polishing algorithm.

        The model is the following trivial formulation:
            minimize
                - sum(x) - sum(y) - 2 sum(z)
            subject to
                sum(x) + 2 sum(y) + 3 sum(z) <= 800
                x, y, z binaries
        """
        # Create variables
        x_vars = [self.NewBoolVar(name=f"x_{i}") for i in range(100)]
        y_vars = [self.NewBoolVar(name=f"y_{i}") for i in range(100)]
        z_vars = [self.NewBoolVar(name=f"z_{i}") for i in range(200)]

        # Set objective
        self.SetObjectiveMin(-sum(x_vars) - sum(y_vars) - 2 * sum(z_vars))

        # Add constraint: x + 2 y + 3 z <= 800
        self.Add(sum(x_vars) + 2 * sum(y_vars) + 3 * sum(z_vars) <= 800, "c0")

        self.Update()

        feasible_sol: pol.VarsSol = {}
        for x in x_vars:
            feasible_sol[self.GetName(x)] = 0.0
        for y in y_vars:
            feasible_sol[self.GetName(y)] = 0.0
        for z in z_vars:
            feasible_sol[self.GetName(z)] = 0.0

        return feasible_sol, x_vars


def solve_toy_example():
    model = MyModel()
    mip_sol, vars_to_fix = model.build_model()

    polish = pol.Polish()
    polish.reset_parameters()
    polish.path = ""
    polish.num_threads = 8
    polish.prefix_log = ""
    polish.prefix_sol = ""
    polish.debug_mode = True
    polish.max_time_no_improve_in_iteration = 30

    pol_sol = polish.polish_solution(
        max_run_time=600,
        init_fixed_perc=0.8,
        init_iter_time_limit=60,
        prob=model,
        init_ms=mip_sol,
        vars_allowed_to_fix=vars_to_fix,
    )
    if not polish.found_valid_solution() or not pol_sol:
        print("No solution found :(")
        exit(0)

    # Uncomment to print the solution's values
    # for var_name, value in pol_sol.items():
    #     print(f"{var_name} = {value}")

    print(f"Obj: {polish.solution.obj}")


if __name__ == "__main__":
    solve_toy_example()
