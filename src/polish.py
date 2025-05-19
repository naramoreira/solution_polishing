"""The main function in this module is Polish.polish_solution, which takes as input a MIP/CP model
and an initial feasible solution, and returns a solution that is at least as good as the warm-up
solution provided by the user.
"""

import logging
import random
import time
from typing import Any, Iterable, NamedTuple

from src.optimiser import Optimiser, OptStatus

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.NOTSET)

logger = logging.getLogger(__name__)

GAP_TOLERANCE = 0.01
"""Optimality tolerance gap for a solution"""

INFEASIBLE_NUM_ITER_TOLERANCE = 5
"""Indicates the tolerance for the number of iterations where the model is found to be
infeasible. If the algorithm reaches INFEASIBLE_NUM_ITER_TOLERANCE + 1 infeasible
iterations, the execution is interrupted and the status is marked as failed.
If the warm-up solution provided by the user is infeasible, the first iteration of
the polishing can select an infeasible subset of variables in the warm-up solution,
causing the model to be infeasible.
Although no infeasibilities are expected in correct implementations, and ideally
this parameter should be set to zero, using a tolerance higher than zero gives the
polish better chances to recover from a bad user solution, from numerical issues or
other unexpected bugs."""

FIX_ONLY_DISCRETE_VARS = True
"""If enabled, the algorithm assumes that the subset of variables allowed to be fixed
during the polishing contains only int variables, and therefore it applies a round
function to the solution's variables values before they are fixed in each iteration.
This helps to avoid numerical issues."""

VarsSol = dict[str, float]
"""Solution type: var_name -> solution_value"""


class PolishStepUpdateCondition(NamedTuple):
    """The "polish step update condition" type gathers different metrics that can be
    combined to define a condition that needs to be met at the end of each polish iteration
    in order to advance a step in the algorithm.

    A step means an update of the percentage of fixed variables (Polish._perc) and/or
    the time limit per iteration (pol_iter_time_limit).

    The combination of the fields max_gap, best_bound_is_zero, max_improve_rate,
    min_num_iter_step and max_abs_improve defines a condition, whereas the fields
    perc_decrease and time_increase defines the consequence if such condition is met.
    More information on how the conditions are applied can be found at
    set_step_update_conditions.

    Args:
        max_gap (float | None): The optimality gap is less or equal to max_gap
            (0 <= max_gap <= 1). This field is ignored if its value is equal to None.
            Defaults to None.
        best_bound_is_zero (bool): The best bound is zero. This condition is mostly useful
            because if the best bound is zero, the optimality gap cannot be properly
            calculated between 0 and 100%. This field is ignored if its value is
            equal to False. Defaults to False.
        max_improve_rate (float | None): The rate in which the solution improved if
            compared to the previous iteration is less or equal to max_improve_rate.
            This field is ignored if its value is equal to None. Defaults to None.
        min_num_iter_step (int): The minimum number of iterations spent at the same
            step before the algorithm can move a step further. This field is ignored
            if its value is less or equal to 1. Defaults to 0.
        max_abs_improve (float | None): The maximum absolute difference between the
            current best solution value and the previous iteration's best solution value
            cannot be higher than max_abs_improve. This field is ignored if its value
            is equal to None. Defaults to None.
        perc_decrease (float): The value to be subtracted from the percentage of fixed
            variables if the condition is met (0 <= Polish._perc <= 1). Defaults to 0.
        time_increase (int): The value to be added to the iteration time limit (seconds)
            if the condition is met. Defaults to 0.


    Example:
        Create a condition to update the step when the optimality gap is below 0.1
        and decrease the percentage of fixed variables by 0.15 while increasing the
        time limit by 10 seconds:

            condition = PolishStepUpdateCondition(
                max_gap=0.1,
                best_bound_is_zero=False,
                max_improve_rate=None,
                min_num_iter_step=0,
                max_abs_improve=None,
                perc_decrease=0.15,
                time_increase=10
            )
    """

    max_gap: float | None = None
    best_bound_is_zero: bool = False
    max_improve_rate: float | None = None
    min_num_iter_step: int = 0
    max_abs_improve: float | None = None
    perc_decrease: float = 0
    time_increase: int = 0

    def __post_init__(self):
        # Step update condition
        assert (
            self.max_gap is not None
            or self.best_bound_is_zero
            or self.max_improve_rate is not None
            or self.min_num_iter_step >= 1
            or self.max_abs_improve is not None
        ), "You must set at least one condition attribute"
        # Consequence if the step update condition passes
        assert (
            self.perc_decrease > 0 or self.time_increase > 0
        ), "You must set a positive value for either perc_decrease or time_increase."


class PolishSolution(NamedTuple):
    """Gathers the information of a solution found by the polish algorithm."""

    sol: VarsSol = {}
    """The solution found by the polish algorithm. It is a dictionary
    where the keys are the variable names and the values are the
    corresponding solution values.
    """
    obj: float | None = None
    """The objective value of the solution found by the polish algorithm.
    """
    bound: float | None = None
    """The best bound of the solution found by the polish algorithm.
    """
    gap: float | None = None
    """The optimality gap of the solution found by the polish algorithm.
    The gap is calculated as the absolute difference between the
    solution value and the best bound, divided by the solution value.
    """
    found_at_iteration: int | None = None
    """The iteration number where the solution was found by the polish algorithm.
    """
    perc_fixed: float | None = None
    """The value of Polish._perc at the time the solution was found by the algorithm.    
    """


class Polish:
    """The Polish class implements the solution polishing algorithm.
    It is used to improve the quality of a given solution by iteratively
    fixing the bounds of a subset of variables and optimizing the modified
    problem. The problem should be a minimization MIP or CP model.

    The algorithm is based on the following steps:
    1. Receives an initial feasible solution to polish.
    2. Fix the bounds of a subset of variables based on their values in the
       best known feasible solution.
    3. Optimize the modified problem.
    4. Update the best solution known.
    5. Repeat steps 2-4 until a stopping condition is met.
    """

    def __init__(self):

        # User parameters

        self.debug_mode = False
        """Indicates if the polish is in debug mode
        """

        self.path = ""
        """The path where output files should be printed (debugging purposes).
        """

        self.prefix_log = ""
        """The prefix to be appended to the polish log messages (debugging purposes).
        """

        self.prefix_sol = ""
        """The prefix to be appended to the name of solutions found by the polish
        (debugging purposes).
        """

        self.user_best_bound: float | None = None
        """Indicates the best bound value for the problem known by the user and which
        is used to try to prove optimality earlier, before reaching the last step of zero
        percentage of fixed variables. Defaults to None.
        """

        self.fix_only_non_zero = False
        """Indicates if only non zero variables values should be fixed along the polishing iterations.
        """

        self.num_threads = 8
        """The number of threads allowed to be used by the optimiser. Default value = 8.
        """

        self.update_step_conditions: list[PolishStepUpdateCondition] = []
        """The conditions used to update the percentage of fixed variables and time per iteration.
        """

        self.max_run_time: int | None = None
        """The time limit (seconds) for the algorithm.
        """

        self.min_iter_runtime: int | None = None
        """Initial time limit per iteration. This value must be a fraction of the
        polish's total time limit.
        """

        self._max_time_no_improve_in_iteration: int | None = None
        """The limit to the continuous time (seconds) spent in an iteration of the
        algorithm without any improvement in the solution value."""

        self._max_total_time_no_improve: int | None = None
        """The limit to the continuous time (seconds) spent in the algorithm
        without any improvement in the solution value."""

        self.epsilon = 0.001
        """The absolute optimality tolerance for a solution. Two solutions will be
        considered the same if abs(solution_1.value - solution_2.value) < epsilon
        """

        self.max_perc_time_find_1_sol = 0.6
        """The maximum fraction of the iteration time limit expected to be spent
        finding the first feasible solution. This must be a float in the range
        [0,1]. Default value = 0.6.

        The time limit per iteration (_iter_time_limit) is automatically increased
        if an iteration spends more than (max_perc_time_find_1_sol * _iter_time_limit)
        seconds to find the first feasible solution.
        """

        # Attributes for internal use

        self._solution = PolishSolution()
        self._log = ""
        self._total_runtime = 0
        self._last_iter_succeed = False
        self._found_infeasibility = False
        self._prob: Optimiser = None
        self._stop_if_perc_zero = False
        self._start_time = 0.0
        self._iter = 0
        self._proved_optimal_using_global_bound = False
        self._proved_optimal_using_user_best_bound = False
        self._free_prob_lp_bound = 0.0
        self._improve_prev = 0.0
        self._improve = 0.0
        self._abs_improve = 0.0
        self._sol_found = False
        self._perc = 0.0
        self._sol_count = 0
        self._succeed = False
        self._iter_time_limit = 0
        self._time_left = 0
        self._num_iter_step = 0
        self._iter_start = 0.0
        self._iter_optimization_start = 0.0
        self._first_sol_time_last_iter = 0.0
        self._first_sol_value = 0.0
        self._time_last_improve = 0.0
        self._algorithm_max_time_no_improve_reached = False
        self._theoretical_iter_time_limit = 0

        # Initialize with default values all entities parameterizable by the user
        self.reset_parameters()

    @property
    def solution(self) -> PolishSolution:
        """Returns the best solution found by the polish algorithm"""
        return self._solution

    @property
    def log(self) -> str:
        """A string with the full polishing log messages, which allows the user
        to see a summary of how the algorithm developed over the iterations.
        It does not include the optimiser log.
        """
        return self._log

    @property
    def total_runtime(self) -> int:
        """Polish total run time"""
        return self._total_runtime

    @property
    def last_iteration_succeeded(self) -> bool:
        """Indicates if the optimizer succeeded finding a feasible solution for the
        last executed iteration. If the last optimization performed was not successful,
        it is not safe to retrieve solution values directly from the optimiser.
        """
        return self._last_iter_succeed

    @property
    def found_infeasibility(self) -> bool:
        """Returns True if the an infeasibility was found while polishing the solution,
        False otherwise.

        Note that infeasibilities are only identified by this attribute if they occur
        more times than the tolerance set by the parameter INFEASIBLE_NUM_ITER_TOLERANCE,
        or if the infeasibility was identified for the full problem while trying to find
        a feasible warm-up solution (in case trust_user_init_sol=False).
        """
        return self._found_infeasibility

    @property
    def max_time_no_improve_in_iteration(self) -> int | None:
        """The limit to the continuous time (seconds) spent in an iteration
        without any improvement in the solution value, or None in case this stopping
        condition is not set."""
        return self._max_time_no_improve_in_iteration

    @max_time_no_improve_in_iteration.setter
    def max_time_no_improve_in_iteration(self, max_iter_time: int):
        """Sets a limit to the continuous time (seconds) spent in an iteration without
        any improvement in the solution value.

        If this time limit is reached, the iteration is terminated and the algorithm
        either moves on to the next iteration or ends the polishing in case of a stopping
        condition for the main loop is met. Exceptionally, if the algorithm is already at
        the last step (percentage of fixed variables is zero) or if the algorithm
        still doesn't know any feasible solution, this stopping condition is not applied.

        Args:
            max_iter_time (int): maximum time (seconds) spent continuously in an
                iteration without improving the solution. Setting it to None disables
                this option (default).
        """
        assert max_iter_time is None or max_iter_time >= 0
        self._max_time_no_improve_in_iteration = max_iter_time

    @property
    def max_total_time_no_improve(self) -> int | None:
        """The limit to the continuous time (seconds) spent in the algorithm
        without any improvement in the solution value, or None in case this stopping
        condition is not set."""
        return self._max_total_time_no_improve

    @max_total_time_no_improve.setter
    def max_total_time_no_improve(self, max_total_time: int):
        """Sets a limit to the continuous time (seconds) spent in the algorithm
        without any improvement in the solution value.

        If this time limit is reached, the current iteration is terminated and
        the algorithm ends the polishing. Exceptionally, if the algorithm still
        doesn't know any feasible solution, this stopping condition is not applied.

        Args:
            max_total_time (int): maximum time (seconds) spent continuously in the
                algorithm without improving the solution. Setting it to None disables
                this option (default).
        """
        assert max_total_time is None or max_total_time >= 0
        self._max_total_time_no_improve = max_total_time

    def _print_log(self, msg: str):
        message = self.prefix_log + ": " + msg
        logger.debug(message)
        self._log += message + "\n"

    def _reset_private_entities(self):
        self._prob = None
        self._sol_found = False
        self._reset_solution()
        self._log = ""
        self._last_iter_succeed = False
        self._free_prob_lp_bound = None
        self._improve_prev = None
        self._improve = None
        self._sol_count = None
        self._first_sol_value = None
        self._succeed = True
        self._found_infeasibility = False
        self._iter = 0
        self._perc = 0
        self._proved_optimal_using_global_bound = False
        self._proved_optimal_using_user_best_bound = False
        self._stop_if_perc_zero = True
        self._num_iter_step = 0
        self._algorithm_max_time_no_improve_reached = False

    def _reset_solution(self):
        self._solution = PolishSolution()

    def _reset_update_conditions(self):
        # Small optimality gap and maximum 30% improvement rate (compared to previous iteration)
        cond1 = PolishStepUpdateCondition(
            max_gap=0.1,
            best_bound_is_zero=False,
            max_improve_rate=0.3,
            max_abs_improve=None,
            perc_decrease=0.15,
            time_increase=10,
        )
        # Bound=0 and maximum 30% improvement rate (compared to previous iteration)
        cond2 = PolishStepUpdateCondition(
            max_gap=None,
            best_bound_is_zero=True,
            max_improve_rate=0.3,
            max_abs_improve=None,
            perc_decrease=0.15,
            time_increase=10,
        )
        # Maximum 50% improvement rate (compared to previous iteration), and maybe higher optimality gap
        cond3 = PolishStepUpdateCondition(
            max_gap=None,
            best_bound_is_zero=False,
            max_improve_rate=0.5,
            max_abs_improve=None,
            perc_decrease=0.1,
            time_increase=5,
        )
        # list of conditions to update the polish fixed percentage and time per iteration
        self.update_step_conditions = [cond1, cond2, cond3]

    def _print_final_log(self):
        logger.info("-- Finished polishing the solution.")
        logger.info(f"-- Runtime {self._total_runtime} seconds.")
        logger.info(f"-- Final fixed percentage: {self._perc:.3f}.")
        logger.info(
            f"-- Best solution value {self._solution.obj} found at iteration "
            f"{self._solution.found_at_iteration} with perc {self._solution.perc_fixed: .3f}."
        )

    def _apply_max_time_no_improve_iter(self) -> bool:
        return self.max_time_no_improve_in_iteration is not None and self._perc != 0

    def _apply_max_time_no_improve_alg(self) -> bool:
        return self.max_total_time_no_improve is not None

    def _time_spent_since_last_improve(self) -> int:
        return int(time.time() - self._time_last_improve)

    def _time_spent_since_iter_start(self) -> int:
        return int(time.time() - self._iter_optimization_start)

    def _set_run_time(self):
        if self._perc <= 0:
            timelimitloc = self._time_left
        else:
            timelimitloc = min(self._iter_time_limit, self._time_left)
            if self._time_left - timelimitloc < self._first_sol_time_last_iter:
                timelimitloc = self._time_left

        self._prob.SetTimeLimit(-timelimitloc)

        # Adjusts time limit if "maximum time without improvement" stopping condition is being used
        self._theoretical_iter_time_limit = abs(self._prob.GetTimeLimit())
        if self._apply_max_time_no_improve_iter():
            self._prob.SetTimeLimit(
                -min(
                    abs(self._prob.GetTimeLimit()),
                    self.max_time_no_improve_in_iteration,
                )
            )

        if self._apply_max_time_no_improve_alg():
            self._prob.SetTimeLimit(
                -min(
                    abs(self._prob.GetTimeLimit()),
                    max(
                        1,
                        self.max_total_time_no_improve
                        - self._time_spent_since_last_improve(),
                    ),
                )
            )

        self._print_log(
            f"Time since last improve = {self._time_spent_since_last_improve()}"
        )
        self._print_log(f"Time limit = {self._prob.GetTimeLimit()}")

    def _get_time_spent(self) -> int:
        return int(time.time() - self._start_time)

    def _set_free_prob_lp_bound(self):
        if self._perc == 0:  # Free-bounds problem
            # First time setting global best bound
            if self._free_prob_lp_bound is not None:
                self._free_prob_lp_bound = max(
                    self._free_prob_lp_bound, self._prob.GetBestBound()
                )
            # Update of global best bound
            else:
                self._free_prob_lp_bound = self._prob.GetBestBound()
        if self._free_prob_lp_bound is not None:
            self._proved_optimal_using_global_bound = (
                abs(self._free_prob_lp_bound - self._solution.obj) < self.epsilon
            )

    def _get_gap(self, solution_value: float, best_bound: float) -> float:
        return (
            abs((solution_value - best_bound) / solution_value)
            if solution_value != 0
            else best_bound
        )

    def _check_user_best_bound(self):
        if (
            self.user_best_bound
            and self._get_gap(self._solution.obj, self.user_best_bound) < GAP_TOLERANCE
        ):
            self._proved_optimal_using_user_best_bound = True
            logger.debug(
                "Polish proved optimal using user best bound = ",
                self.user_best_bound,
            )

    def _get_iter_sol_improve(self) -> float:
        last_sol_obj = (
            self._solution.obj
            if self._solution.found_at_iteration is not None
            else self._first_sol_value
        )
        improve = last_sol_obj - self._prob.GetObjValue()
        if abs(improve) < self.epsilon:  # Avoid numerical issues
            improve = 0
        if last_sol_obj != 0:
            improve = abs(improve / last_sol_obj)
        assert improve >= 0
        return improve

    def _check_time_stop_cond(self):
        if (
            self._apply_max_time_no_improve_alg()
            and self._time_spent_since_last_improve() + 1
            >= self.max_total_time_no_improve
        ):
            self._algorithm_max_time_no_improve_reached = True
            self._print_log(
                "Algorithm's stopping condition reached: maximum time with no "
                f"improvement ({self._time_spent_since_last_improve()})"
            )
        elif (
            self._apply_max_time_no_improve_iter()
            and self._prob.GetStatus() == OptStatus.INFEASIBLE
            and int(time.time() - self._iter_optimization_start)
            < self._theoretical_iter_time_limit
        ):
            time_no_improvement = int(
                time.time()
                - max(self._iter_optimization_start, self._time_last_improve)
            )
            self._print_log(
                "Iteration's stopping condition reached: maximum time with no "
                f"improvement ({time_no_improvement})"
            )

    def _process_solution(self):
        self._num_iter_step += 1
        self._sol_count = self._prob.GetNumSols()
        self._sol_found = True if self._sol_count > 0 else False
        self._last_iter_succeed = self._sol_found
        self._abs_improve = 0
        if self._prob.GetStatus() == OptStatus.INFEASIBLE:
            if self._iter == 0 or self._iter > INFEASIBLE_NUM_ITER_TOLERANCE:
                self._print_log(
                    "Error: while polishing solution, problem is infeasible"
                )
                self._succeed = False
                self._found_infeasibility = True

        self._total_runtime = self._get_time_spent()
        if self._sol_found:
            if self._iter == 1:
                self._first_sol_value = self._prob.sol_callback.value_1st_sol
            self._succeed = True
            self._improve_prev = self._improve
            self._improve = self._get_iter_sol_improve()

            if self._improve > 0 or self._solution.found_at_iteration is None:
                self._time_last_improve = time.time()
                prev_value = (
                    self._first_sol_value if self._iter <= 1 else self._solution.obj
                )
                self._abs_improve = abs(prev_value - self._prob.GetObjValue())

                self._solution = PolishSolution(
                    sol=self._prob.GetSolution(),
                    obj=self._prob.GetObjValue(),
                    bound=self._prob.GetBestBound(),
                    gap=self._get_gap(
                        self._prob.GetObjValue(), self._prob.GetBestBound()
                    ),
                    found_at_iteration=self._iter,
                    perc_fixed=self._perc,
                )

                improve_rate = self._improve / (
                    self._improve_prev
                    if self._improve_prev is not None and self._improve_prev > 0
                    else self._improve
                )
                self._print_log(
                    f"Solution improved to {self._solution.obj} at iteration "
                    f"{self._iter} with perc={self._perc:.3f}. "
                    f"Iteration's optimality gap = {round(self._solution.gap * 10000) / 100}%"
                    f", Solution improve = {100 * self._improve}%, Improve rate = {improve_rate}"
                )
            elif self._improve < 0:
                self._print_log(
                    f"Warning: solution has worsened at iteration {self._iter}"
                    f". Best solution = {self._solution.obj}"
                    f", new solution = {self._prob.GetObjValue()}"
                    ". Please check if the user solution was loaded correctly."
                )
            else:
                self._print_log(
                    f"No improvements at iteration {self._iter} with perc={self._perc:.3f}."
                    f" Best known solution {self._solution.obj} "
                    f"found at iteration {self._solution.found_at_iteration} "
                    f"with perc={self._solution.perc_fixed:.3f}."
                )

            if self._iter == 1 and self.debug_mode:
                self._print_log(f"First solution's value = {self._first_sol_value}")

            self._set_free_prob_lp_bound()
            self._check_user_best_bound()
        else:
            # Simulate "not enough time"
            self._first_sol_time_last_iter = abs(self._prob.GetTimeLimit()) + 1
            self._print_log("No polished solution found.")
        self._check_time_stop_cond()

    def _passes(self, condition: PolishStepUpdateCondition) -> bool:
        max_gap_condition_satisfied = (
            condition.max_gap is None
            or self._get_gap(self._prob.GetObjValue(), self._prob.GetBestBound())
            < condition.max_gap
        )
        return (
            self._sol_found
            and max_gap_condition_satisfied
            and (self._prob.GetBestBound() == 0 or not condition.best_bound_is_zero)
            and (self._num_iter_step >= condition.min_num_iter_step)
            and (
                condition.max_abs_improve is None
                or self._abs_improve <= condition.max_abs_improve
            )
            and (
                condition.max_improve_rate is None
                or (
                    self._improve_prev is not None
                    and self._improve <= condition.max_improve_rate * self._improve_prev
                )
            )
        )

    def _update_perc(self):
        cnt = 0
        for condition in self.update_step_conditions:
            cnt += 1
            if self._passes(condition) or self._perc <= 0:
                self._perc = self._perc - condition.perc_decrease
                self._iter_time_limit += condition.time_increase
                self._num_iter_step = 0
                self._print_log(f"Polish step updated by condition {cnt}")
                break  # Consider the first valid condition found

        self._perc = max(self._perc, 0)
        self._perc = min(self._perc, 1)

        temp = self._iter_time_limit
        self._iter_time_limit = max(
            self._iter_time_limit,
            round(self._first_sol_time_last_iter / self.max_perc_time_find_1_sol),
        )
        if temp != self._iter_time_limit:
            self._print_log(
                "Time per iteration increased using condition max_perc_time_find_1_sol. "
                f"Iteration time limit changed from {temp} to {self._iter_time_limit}"
            )

    def _find_init_solution(self):
        self._print_log("----- Finding an initial solution to polish")

        # Save original parameters
        max_mip_sol = self._prob.GetSolutionCountLimit()

        # Tries loading the initial solution provided by the user
        if len(self._solution.sol) > 0:
            self._prob.AddFullHints(
                vars_sol=self._solution.sol,
            )

        # Finds an initial int solution
        self._prob.SetSolutionCountLimit(1)

        self._prob.Optimise()

        # Update solution
        self._process_solution()
        if not self._sol_found:
            self._print_log("No initial solution found.")

        self._prob.SetSolutionCountLimit(max_mip_sol)

    def _config_logging(self):
        file_handler = logging.FileHandler("polish.log", mode="w")
        file_handler.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s:%(levelname)s:%(message)s", datefmt="%Y/%m/%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def found_valid_solution(self):
        """Returns True if the algorithm found a feasible solution to the problem, False otherwise."""
        return self._solution.found_at_iteration is not None

    def reached_original_problem(self):
        """Returns True if the algorithm reached the last step (percentage of fixed
        variables = 0), which means the last MIP solved was the full problem.
        Otherwise, if any other stopping condition was reached before the percentage
        of fixed variables is set to zero, returns False.
        """
        return self._perc == 0

    def set_step_update_conditions(
        self, user_conditions: list[PolishStepUpdateCondition]
    ):
        """Replaces the default conditions used to update the percentage of fixed variables and time per iteration by conditions customised by the user.

        A condition is defined using the type PolishStepUpdateCondition.
        The conditions to update the polish step are checked after the optimization at the end of each iteration,
        following their order in the list. No more than one condition can take effect per iteration, which means
        that once a condition is met, it is applied and the next ones will not be checked. Therefore, the conditions
        in the list should be sorted according to their priority.

        Args:
            user_conditions (list[PolishStepUpdateCondition]): list of conditions to update the polish step
        """
        if len(user_conditions) != 0:
            self._print_log(
                "Applied user conditions for updating the polish's steps parameters."
            )
            self.update_step_conditions = user_conditions

    def reset_parameters(self):
        """Resets to the default values all the polish parameters available to the user to tune the algorithm.

        To ensure you don't have undesired changes in the parameters settings, residual from previous calls
        to polish_solution, it is a good practice to call pol_reset_parameters just before
        any changes to the default parameters values and just: call polish_solution.
        """
        self.user_best_bound = None
        self.fix_only_non_zero = False
        self.path = ""
        self.num_threads = 8
        self.prefix_log = ""
        self.prefix_sol = ""
        self.debug_mode = False
        self._reset_update_conditions()
        self.max_time_no_improve_in_iteration = None
        self.max_total_time_no_improve = None
        self.epsilon = 0.001
        self.max_perc_time_find_1_sol = 0.6

    def polish_solution(
        self,
        max_run_time: int,
        init_fixed_perc: float,
        init_iter_time_limit: int,
        prob: Optimiser,
        init_ms: VarsSol,
        vars_allowed_to_fix: Iterable[Any],
    ) -> VarsSol | None:
        """Runs the solution polishing algorithm to solve a minimization MIP/CP problem and
        returns a solution that is at least as good as the warm-up solution provided by the user.

        The solution polishing algorithm is a local search heuristic based on an iterative approach.
        At each iteration, it fixes the bounds of a subset of variables based an their values
        in the best known feasible solution, optimizes the modified model (minimization), updates
        the best solution known, and releases the bounds to their original values.

        The selection of the subset of variables to be fixed is done randomly, so that at each
        iteration a different feasible model is solved.

        The percentage of variables being fixed should generally decrease along the iterations,
        until the point that either a stopping condition is met or 0% of variables is fixed and
        the problem is therefore the original one.

        Args:
            max_run_time (int): the time limit (seconds) for the algorithm
            init_fixed_perc (float): the initial ratio of variables from the subset
                vars_allowed_to_fix to be fixed at the first iteration.
                The value must be in the range [0,1].
            init_iter_time_limit (int): initial time limit per iteration.
                This value must be a fraction of maxRunTime.
            prob (Optimiser): the MIP/CP problem
            init_ms (VarsSol): an objective containing an initial feasible solution.
                If init_ms is empty ({}), the algorithm will try to find an initial
                feasible solution by making an extra call to the optimizer prior to
                start the polishing iterations just to find an initial feasible solution
                for the full problem.
            vars_allowed_to_fix (Set[Any]): the subset of variables in the problem
                whose bounds can be fixed by the algorithm.

        Returns:
            VarsSol | None: object with the best solution found by the end of the polishing.
        """

        self._config_logging()

        if init_iter_time_limit > max_run_time:
            self._print_log(
                "Warning: not enough time to run the polish. The time limit provided "
                f"({max_run_time}) is lower than the minimum time estimated per iteration"
                f" ({init_iter_time_limit}). Returning..."
            )
            return None

        self._print_log(f"Polish called with total time limit = {max_run_time}")

        random.seed(1)

        self._reset_private_entities()
        self._prob = prob
        self._start_time = time.time()
        self._total_runtime = 0
        self._solution = PolishSolution(sol=init_ms)
        self.max_run_time = abs(max_run_time)
        self._iter_time_limit = abs(init_iter_time_limit)
        self.min_iter_runtime = abs(init_iter_time_limit)
        self._first_sol_time_last_iter = self.min_iter_runtime

        self._prob.SetLogOptimiser(self.debug_mode)
        self._prob.SetNumThreads(self.num_threads)
        self._prob.TurnOffHeuristics()
        self._prob.SetTimeLimit(-self.max_run_time)

        if not init_ms:
            self._find_init_solution()  # ensures an initial solution
            if self._sol_found:
                assert self._solution.gap is not None
                if self._solution.gap <= GAP_TOLERANCE:
                    self._print_final_log()
                    return self._solution.sol

        vars_list = []
        var_lb_values: list[float] = []
        var_ub_values: list[float] = []
        for v in vars_allowed_to_fix:
            vars_list.append(v)
            var_lb_values.append(self._prob.GetLB(v))
            var_ub_values.append(self._prob.GetUB(v))

        self._perc = init_fixed_perc
        self._time_left = self.max_run_time - self._total_runtime
        stop_condition = self._time_left <= 0
        self._time_last_improve = time.time()

        # Start polish iterations
        while not stop_condition:
            self._stop_if_perc_zero = True
            self._iter_start = time.time()
            self._iter += 1
            self._print_log(f"Total time left to polish = {self._time_left}")
            self._print_log(
                f"----- Starting polish iteration {self._iter} with perc = {self._perc:.3f}"
            )

            # Add warm-up solution
            self._prob.AddFullHints(
                vars_sol=self._solution.sol,
            )

            # Randomly choose perc of solution to fix variables values in the MIP according to the current best solution
            fixed_vars = []
            fixed_values = []
            for v in vars_allowed_to_fix:
                varvalue = self._solution.sol[self._prob.GetName(v)]
                if FIX_ONLY_DISCRETE_VARS:
                    varvalue = round(varvalue)
                if random.random() < self._perc and (
                    not self.fix_only_non_zero or varvalue != 0
                ):
                    fixed_vars.append(v)
                    fixed_values.append(varvalue)

            bnd_type = ["B"] * len(fixed_vars)
            self._prob.ChangeBounds(fixed_vars, bnd_type, fixed_values)

            self._set_run_time()

            # Loosen mip optimality tolerance if variables are being fixed
            if self._perc > 0:
                original_mip_rel_stop = self._prob.GetMipRelStop()
                self._prob.SetMipRelStop(0.01)

            # Optimize MIP (minimization)
            self._iter_optimization_start = time.time()
            callback = self._prob.GetNewCallback(
                time_limit_no_improve=(
                    self.max_time_no_improve_in_iteration
                    if self._apply_max_time_no_improve_iter()
                    else None
                )
            )
            self._prob.Optimise(callback)

            # Reset original mip optimality tolerance
            if self._perc > 0:
                self._prob.SetMipRelStop(original_mip_rel_stop)

            # Load solution
            self._process_solution()

            self._time_left = self.max_run_time - self._total_runtime

            stop_condition = (
                self._time_left < self._first_sol_time_last_iter
                or (self._stop_if_perc_zero and self._perc <= 0)
                or not self._succeed
                or self._proved_optimal_using_global_bound
                or self._proved_optimal_using_user_best_bound
                or self._algorithm_max_time_no_improve_reached
            )
            if not stop_condition:
                self._update_perc()

            # Restore bounds of fixed variables
            self._prob.ChangeBounds(vars_list, ["L"] * len(vars_list), var_lb_values)
            self._prob.ChangeBounds(vars_list, ["U"] * len(vars_list), var_ub_values)

        self._print_final_log()
        return self._solution.sol
