import logging
import math
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Timer
from typing import Any, Dict, List, Optional, Union

import gurobipy as gp
import ortools
from ortools.sat.python import cp_model


class OptimiserSoftware(Enum):
    GUROBI = 1
    ORTOOLS_CPSAT = 2


class OptStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    NO_SOLUTION_FOUND = "NO_SOLUTION_FOUND"
    UNSET = "UNSET"


@dataclass
class OptTimer:

    model: "Optimiser"
    total_time_limit: int
    _total_timer: Optional[Timer] = None

    time_limit_no_improve: Optional[int] = None
    _timer_no_improve: Optional[Timer] = None

    def __post_init__(self):
        self._reset_timer()
        self._reset_total_timer()

    def cancel_timer(self):
        if self._timer_no_improve:
            self._timer_no_improve.cancel()

    def _reset_timer(self):
        self.cancel_timer()
        if self.time_limit_no_improve:
            self._timer_no_improve = Timer(self.time_limit_no_improve, self.stop_search)
            self._timer_no_improve.start()

    def stop_search(self):
        logging.info(
            f"Forcing stop after {self.time_limit_no_improve} seconds without improvement"
        )
        self.model.model.terminate()

    def cancel_total_timer(self):
        if self._total_timer:
            self._total_timer.cancel()

    def _reset_total_timer(self):
        self.cancel_total_timer()
        self._total_timer = Timer(self.total_time_limit, self.stop_search_by_total_time)
        self._total_timer.start()

    def stop_search_by_total_time(self):
        logging.info(
            f"Forcing stop after reaching the time limit of {self.total_time_limit} seconds "
        )
        self.model.model.terminate()


class NewSolGurobiCallback:
    """
    Retrieve intermediate solutions.
    """

    user_update_func: Optional[Callable] = None
    solution_count_limit: Optional[int] = None

    _solution_count: int = 0
    _time_1st_sol: Optional[int] = None
    _value_1st_sol: Optional[float] = None

    timer: OptTimer

    def __init__(
        self,
        time_limit: int,
        model: "Optimiser",
        solution_count_limit: Optional[int] = None,
        time_limit_no_improve: Optional[int] = None,
        user_update: Optional[Callable] = None,
    ):
        NewSolGurobiCallback._solution_count = 0
        NewSolGurobiCallback._time_1st_sol = None
        NewSolGurobiCallback._value_1st_sol = None
        NewSolGurobiCallback.solution_count_limit = solution_count_limit
        NewSolGurobiCallback.user_update_func = user_update
        NewSolGurobiCallback.timer = OptTimer(
            total_time_limit=time_limit,
            time_limit_no_improve=time_limit_no_improve,
            model=model,
        )

    @property
    def solution_count(self) -> int:
        return NewSolGurobiCallback._solution_count

    @property
    def time_1st_sol(self) -> Union[int, None]:
        return NewSolGurobiCallback._time_1st_sol

    @property
    def value_1st_sol(self) -> Union[float, None]:
        return NewSolGurobiCallback._value_1st_sol

    @staticmethod
    def callback(model, where):
        try:
            if where == gp.GRB.Callback.MIPSOL:
                NewSolGurobiCallback.timer._reset_timer()
                objbst = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST)
                objbnd = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)
                objsols = (
                    model.cbGet(gp.GRB.Callback.MIPSOL_SOLCNT) + 1
                )  # Gurobi only updates MIPSOL_SOLCNT after the callback
                objtime = model.cbGet(gp.GRB.Callback.RUNTIME)

                NewSolGurobiCallback._solution_count += 1
                if objsols == 1 and NewSolGurobiCallback._time_1st_sol == None:
                    NewSolGurobiCallback._time_1st_sol = objtime
                    NewSolGurobiCallback._value_1st_sol = objbst

                if abs(objbnd) == 1e100 and objsols == 1:
                    # if there is still no value known for the best bound (e.g. user solution is the only being loaded), set to zero
                    objbnd = 0

                NewSolGurobiCallback._solution_count = objsols
                if NewSolGurobiCallback.user_update_func:
                    NewSolGurobiCallback.user_update_func(
                        objbst=objbst,
                        objbnd=objbnd,
                        objsols=objsols,
                        objtime=objtime,
                        getSolution=model.cbGetSolution,
                    )
                logging.info(
                    f"Solution {objsols}, time = {objtime} s, objective = {objbst}"
                )
                if (
                    NewSolGurobiCallback.solution_count_limit is not None
                    and objsols >= NewSolGurobiCallback.solution_count_limit
                ):
                    logging.info(f"Stop search after {objsols} solutions")
                    model.terminate()
        except Exception as e:
            import traceback

            traceback.print_exc()
            logging.info(e)
            raise e


class NewSolCPSatCallBack(cp_model.CpSolverSolutionCallback):
    """
    Retrieve intermediate solutions and allow interruption of the optimiser
    in case a long enough time is spent without any improvements to the solution.
    """

    def __init__(
        self,
        time_limit: int,
        solution_count_limit: Union[int, None] = None,
        time_limit_no_improve: Union[int, None] = None,
    ):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._solution_count = 0
        self._solution_count_limit = solution_count_limit
        self._timer_limit = time_limit_no_improve
        self._timer = None
        self._time_1st_sol = None
        self._value_1st_sol: Optional[float] = None
        self._reset_timer()
        self._total_timer = None
        self._total_time_limit = time_limit
        self._reset_total_timer()

    @property
    def solution_count(self) -> int:
        return self._solution_count

    @property
    def time_1st_sol(self) -> Union[int, None]:
        return self._time_1st_sol

    @property
    def value_1st_sol(self) -> Union[float, None]:
        return self._value_1st_sol

    def on_solution_callback(self):
        try:
            self._reset_timer()
            self._solution_count += 1
            if self._solution_count == 1:
                self._time_1st_sol = self.WallTime()
                self._value_1st_sol = self.ObjectiveValue()
            self.user_update()
            logging.info(
                f"Solution {self._solution_count}, time = {self.WallTime()} s, objective = {self.ObjectiveValue()}"
            )
            if (
                self._solution_count_limit is not None
                and self._solution_count >= self._solution_count_limit
            ):
                logging.info(
                    f"Stop search after {self._solution_count_limit} solutions"
                )
                super().StopSearch()
        except Exception as e:
            traceback.print_exc()
            logging.error(e)
            raise e

    def user_update(self):
        pass

    def cancel_timer(self):
        if self._timer:
            self._timer.cancel()

    def _reset_timer(self):
        self.cancel_timer()
        if self._timer_limit:
            self._timer = Timer(self._timer_limit, self.stop_search)
            self._timer.start()

    def stop_search(self):
        logging.info(
            f"Forcing stop after {self._timer_limit} seconds without improvement"
        )
        super().StopSearch()

    def cancel_total_timer(self):
        if self._total_timer:
            self._total_timer.cancel()

    def _reset_total_timer(self):
        self.cancel_total_timer()
        self._total_timer = Timer(
            self._total_time_limit, self.stop_search_by_total_time
        )
        self._total_timer.start()

    def stop_search_by_total_time(self):
        logging.info(
            f"Forcing stop after reaching the time limit of {self._total_time_limit} seconds "
        )
        super().StopSearch()


class VarType(Enum):
    Integer = 1
    Continuous = 2

    def fractional(self) -> bool:
        return self == VarType.Continuous


class Optimiser:
    """
    This class works as an interface to facilitate the creation of mathematical
    models by using a single syntax regardless of the optimiser software used.

    The user can then create a problem-specific model class that inherits from
    Optimiser, and only use the functions available in Optimiser to access any
    feature of a specific optimiser software.

    The current options of optimisers are listed by OptimiserSoftware. Because different optimisers
    offer different features, some functions are available only for some optimisers.
    """

    def __init__(self, optimiser: OptimiserSoftware):
        self.optimiser: OptimiserSoftware = optimiser
        self.solver_status = None
        self.sol_callback: Union[NewSolGurobiCallback, NewSolCPSatCallBack, None] = None

        if self.optimiser == OptimiserSoftware.GUROBI:
            self.model = gp.Model()
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            self.model = cp_model.CpModel()
            self.solver = cp_model.CpSolver()
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def Add(self, constraint, name: str = "", lazy: int = 0):
        if self.optimiser == OptimiserSoftware.GUROBI:
            constr = self.model.addConstr(constraint, name=name)
            constr.Lazy = lazy
            return constr
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            return self.model.Add(constraint)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def AddWithOnlyEnforceIf(self, constraint, only_enforce_if_var, name: str = ""):
        if self.optimiser == OptimiserSoftware.GUROBI:
            self.model.addGenConstrIndicator(
                only_enforce_if_var, True, constraint, name=name
            )
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            self.model.Add(constraint).OnlyEnforceIf(only_enforce_if_var)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def AddWithOnlyEnforceIfNot(self, constraint, only_enforce_if_var, name: str = ""):
        if self.optimiser == OptimiserSoftware.GUROBI:
            self.model.addGenConstrIndicator(
                only_enforce_if_var, False, constraint, name=name
            )
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            self.model.Add(constraint).OnlyEnforceIf(self.BoolNot(only_enforce_if_var))
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def AddSOS(self, vars_list, weights=None, sos_type=gp.GRB.SOS_TYPE1):
        """
        gp.GRB.SOS_TYPE1: at most 1 variable in vars_list can be non-zero
        gp.GRB.SOS_TYPE2: at most 2 variables in vars_list can be non-zero and the non-zero
        values must be contiguous
        """
        if self.optimiser == OptimiserSoftware.GUROBI:
            self.model.addSOS(type=sos_type, vars=vars_list, wts=weights)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def BoolNot(self, variable):
        if isinstance(variable, (int, bool)):
            return not variable
        elif self.optimiser == OptimiserSoftware.GUROBI:
            return 1 - variable
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            return variable.Not()
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def NewBoolVar(self, name: str):
        if self.optimiser == OptimiserSoftware.GUROBI:
            return self.model.addVar(vtype=gp.GRB.BINARY, name=name)
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            return self.model.NewBoolVar(name)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def NewIntVar(self, lb, ub, name):
        if lb > ub:
            raise ValueError(f"Lower bound {lb} cannot be higher than upper bound {ub}")
        lb = math.ceil(lb)
        ub = math.floor(ub)
        if self.optimiser == OptimiserSoftware.GUROBI:
            return self.model.addVar(lb=lb, ub=ub, vtype=gp.GRB.INTEGER, name=name)
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            return self.model.NewIntVar(lb, ub, name)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def NewContVar(self, lb, ub, name):
        if lb > ub:
            raise ValueError(f"Lower bound {lb} cannot be higher than upper bound {ub}")
        if self.optimiser == OptimiserSoftware.GUROBI:
            return self.model.addVar(lb=lb, ub=ub, vtype=gp.GRB.CONTINUOUS, name=name)
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            return self.model.NewIntVar(
                lb, ub, name
            )  # CP-SAT only accepts integer variables
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def NewIntervalVar(self, start, size, end, name):
        """
        Creates an interval variable from start, size, and end.
        An interval variable is a constraint, that is itself used in other constraints like NoOverlap.
        Internally, it ensures that start + size == end
        """
        if self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            return self.model.NewIntervalVar(start, size, end, name)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def NewOptionalIntervalVar(self, start, size, end, is_present, name):
        """
        Creates an optional interval var from start, size, end, and is_present.
        An optional interval variable is a constraint, that is itself used in other constraints like NoOverlap.
        This constraint is protected by an is_present literal that indicates if it is active or not.
        Internally, it ensures that is_present implies start + size == end
        """
        if self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            return self.model.NewOptionalIntervalVar(start, size, end, is_present, name)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def AddCumulative(self, intervals, demands, capacity):
        """
        intervals: The list of interval variables NewIntervalVar.
        demands: The list of demands for each interval. Each demand must be >= 0. Each demand can be an integer value, or an integer variable.
        capacity: The maximum capacity of the cumulative constraint. It must be a positive integer value or variable.
        """
        if self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            return self.model.AddCumulative(intervals, demands, capacity)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def AddMaxEquality(self, target, variables):
        if self.optimiser == OptimiserSoftware.GUROBI:
            for v in variables:
                self.model.addConstr(target >= v)
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            self.model.AddMaxEquality(target, variables)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def AddMinEquality(self, target, variables):
        if self.optimiser == OptimiserSoftware.GUROBI:
            for v in variables:
                self.model.addConstr(target <= v)
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            self.model.AddMinEquality(target, variables)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def SetObjectiveMin(self, obj, index=None, priority=0, name=""):
        if self.optimiser == OptimiserSoftware.GUROBI:
            if index == None:
                self.model.setObjective(obj, gp.GRB.MINIMIZE)
            else:
                self.model.setObjectiveN(obj, index=index, priority=priority, name=name)
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            self.model.Minimize(obj)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def AddHint(self, var, value):
        if self.optimiser == OptimiserSoftware.GUROBI:
            var.Start = value
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            self.model.AddHint(var=var, value=value)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def AddFullHints(self, vars_sol):
        if self.optimiser == OptimiserSoftware.GUROBI:
            self._add_full_hints_gurobi(vars_sol)
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            self._add_full_hints_ortools(vars_sol)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def _add_full_hints_gurobi(self, vars_sol):
        if len(vars_sol) == 0:
            return
        logging.info("Adding hints...")
        for i, var in enumerate(self.model.getVars()):
            if var.varname in vars_sol:
                var.Start = vars_sol[var.varname]
            else:
                logging.warning(f"Missing hint value for variable {var.varname}")

    def _add_full_hints_ortools(self, vars_sol):
        self.model.ClearHints()
        self.solver.parameters.keep_all_feasible_solutions_in_presolve = True
        if len(vars_sol) == 0:
            return
        logging.info("Adding hints...")
        for i, var in enumerate(self.model.Proto().variables):
            if var.name in vars_sol:
                self.model.Proto().solution_hint.vars.append(i)
                value_int = round(vars_sol[var.name])
                value = vars_sol[var.name]
                if value_int != value:
                    raise TypeError(
                        f"CP-SAT variables can only be assigned to integer values. "
                        f"The user warm-up solution has value = {value} for variable {var.name}"
                    )
                self.model.Proto().solution_hint.values.append(value_int)
            else:
                logging.warning(f"Missing hint value for variable {var.name}")

    def ChangeRandomSeed(self):
        if self.optimiser == OptimiserSoftware.GUROBI:
            self._print_not_implemented_msg(method="ChangeRandomSeed")
            return
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            self.solver.parameters.random_seed += 10  # any change to random seed
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def Update(self):
        if self.optimiser == OptimiserSoftware.GUROBI:
            self.model.update()
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            self._print_not_implemented_msg(method="Update", warning=False)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def GetName(self, var) -> str:
        if self.optimiser == OptimiserSoftware.GUROBI:
            return var.varname
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            if isinstance(var, ortools.sat.cp_model_pb2.IntegerVariableProto):
                return var.name
            elif isinstance(var, cp_model.IntVar):
                return var.Name()
            raise Exception(f"Error: variable {var} could not be found.")
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def GetVar(self, var_name) -> Any:
        if self.optimiser == OptimiserSoftware.GUROBI:
            for var in self.model.getVars():
                if var.varname == var_name:
                    return var
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            for var in self.model.Proto().variables:
                if var.name == var_name:
                    return var
        else:
            raise UnsupportedOptimiserException(self.optimiser)
        logging.warning(
            f"Warning: variable with name {var_name} could not be found. Returning None."
        )
        return None

    def GetType(self, var) -> VarType:
        if self.optimiser == OptimiserSoftware.GUROBI:
            if var.VType == "C" or var.VType == "S":
                return VarType.Continuous
            if var.VType == "B" or var.VType == "I" or var.VType == "N":
                return VarType.Integer
            raise Exception(f"Unknown variable type {var.VType} for Gurobi")
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            return VarType.Integer
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def GetUB(self, var) -> float:
        if self.optimiser == OptimiserSoftware.GUROBI:
            return var.UB
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            if isinstance(var, ortools.sat.cp_model_pb2.IntegerVariableProto):
                return var.domain[-1]
            elif isinstance(var, cp_model.IntVar):
                return var.Proto().domain[-1]
        raise UnsupportedOptimiserException(self.optimiser)

    def GetLB(self, var) -> float:
        if self.optimiser == OptimiserSoftware.GUROBI:
            return var.LB
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            if isinstance(var, ortools.sat.cp_model_pb2.IntegerVariableProto):
                return var.domain[0]
            elif isinstance(var, cp_model.IntVar):
                return var.Proto().domain[0]
        raise UnsupportedOptimiserException(self.optimiser)

    def GetObjValue(self) -> Union[float, int]:
        if self.optimiser == OptimiserSoftware.GUROBI:
            return self.model.ObjVal
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            return self.solver.ObjectiveValue()
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def GetBestBound(self) -> Union[float, int]:
        if self.optimiser == OptimiserSoftware.GUROBI:
            return self.model.ObjBound
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            return self.solver.BestObjectiveBound()
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def GetValue(self, input) -> Union[float, int]:
        if self.optimiser == OptimiserSoftware.GUROBI:
            if isinstance(input, gp.LinExpr):
                return input.getValue()
            elif isinstance(input, gp.Var):
                return input.X
            else:
                return input
        else:
            return self.solver.Value(input)

    def GetSolution(self, include_zeros=True) -> dict:
        # Retrieves full incumbent solution
        if self.optimiser == OptimiserSoftware.GUROBI:
            return self._GetSolDictGurobi(include_zeros)
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            return self._GetSolDictORTools(include_zeros)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def _GetSolDictGurobi(self, include_zeros):
        vars_sol = {}
        vars = self.model.getVars()
        sol = self.model.getAttr(gp.GRB.Attr.X, vars)
        for i, var in enumerate(vars):
            value = sol[i]
            if not self.GetType(var).fractional():
                value = round(value)  # Avoid numerical issues
            if include_zeros or value != 0:
                vars_sol[var.varname] = value
        return vars_sol

    def _GetSolDictORTools(self, include_zeros):
        vars_sol = {}
        for i, var in enumerate(self.model.Proto().variables):
            value = self.solver.ResponseProto().solution[i]
            if include_zeros or value != 0:
                vars_sol[var.name] = value
        return vars_sol

    def GetVars(self) -> Dict[str, Any]:
        # Retrieves all variables in the model
        if self.optimiser == OptimiserSoftware.GUROBI:
            return self._GetVarsDictGurobi()
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            return self._GetVarsDictORTools()
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def _GetVarsDictGurobi(self) -> Dict[str, Any]:
        vars_sol = {}
        for i, var in enumerate(self.model.getVars()):
            vars_sol[var.varname] = var
        return vars_sol

    def _GetVarsDictORTools(self) -> Dict[str, Any]:
        vars_sol = {}
        for i, var in enumerate(self.model.Proto().variables):
            vars_sol[var.name] = var
        return vars_sol

    def AddStrategy(self, vars_list, var_strategy, domain_strategy):
        if self.optimiser == OptimiserSoftware.GUROBI:
            self._print_not_implemented_msg(method="AddStrategy")
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            self.model.AddDecisionStrategy(vars_list, var_strategy, domain_strategy)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def FixValue(self, value: Union[float, int, bool], var):
        """
        Auxiliary function to debug if the initial solution provided to the model is feasible.
        Suggestion: call it inside Model.add_full_hints so that the initial solution is fixed.
        """
        if isinstance(var, str):
            var = self.GetVar(var)
            if not var:
                return
        if isinstance(value, bool):
            self.Add(var == int(value))
        else:
            self.Add(var == value)

    def ChangeBounds(
        self,
        fixed_vars: List[Any],
        bnd_type: List[str],
        fixed_values: List[float],
    ):
        assert len(fixed_vars) == len(bnd_type) == len(fixed_values)
        if self.optimiser == OptimiserSoftware.GUROBI:
            for i, v in enumerate(fixed_vars):
                if bnd_type[i] == "L":
                    v.LB = fixed_values[i]
                elif bnd_type[i] == "U":
                    v.UB = fixed_values[i]
                elif bnd_type[i] == "B":
                    v.LB = fixed_values[i]
                    v.UB = fixed_values[i]
        if self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            for i, v in enumerate(fixed_vars):
                if bnd_type[i] == "L":
                    ub = self.GetUB(v)
                    self.ChangeBound(v, fixed_values[i], ub)
                elif bnd_type[i] == "U":
                    lb = self.GetLB(v)
                    self.ChangeBound(v, lb, fixed_values[i])
                elif bnd_type[i] == "B":
                    self.ChangeBound(v, fixed_values[i], fixed_values[i])
        else:
            raise Exception(
                f"Method ChangeBounds not implement for opt software {self.optimiser}"
            )

    def ChangeBound(
        self,
        var: Any,
        lower_bound: float,
        upper_bound: float,
    ):
        if self.optimiser == OptimiserSoftware.GUROBI:
            var.LB = lower_bound
            var.UB = upper_bound
        if self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            if isinstance(var, ortools.sat.cp_model_pb2.IntegerVariableProto):
                var.domain[:] = []
                var.domain.extend(
                    cp_model.Domain(lower_bound, upper_bound).FlattenedIntervals()
                )
            elif isinstance(var, cp_model.IntVar):
                var.Proto().domain[:] = []
                var.Proto().domain.extend(
                    cp_model.Domain(lower_bound, upper_bound).FlattenedIntervals()
                )
        else:
            raise Exception(
                f"Method ChangeBound not implement for opti software {self.optimiser}"
            )

    def GetSolutionCountLimit(self) -> Optional[int]:
        if self.optimiser == OptimiserSoftware.GUROBI:
            return self.model.Params.SolutionLimit
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            if self.sol_callback is not None:
                assert isinstance(self.sol_callback, NewSolCPSatCallBack)
                return self.sol_callback._solution_count_limit
            return None
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def SetSolutionCountLimit(self, value: Optional[int]):
        if value is None:
            return
        if self.optimiser == OptimiserSoftware.GUROBI:
            self.model.setParam(gp.GRB.Param.SolutionLimit, value)
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            msg = "Please set the solution count limit in the callback when calling the optimiser"
            self._print_not_implemented_msg(
                method="SetSolutionCountLimit", extra_msg=msg
            )
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def SetMIPGap(self, value: float):
        if self.optimiser == OptimiserSoftware.GUROBI:
            self.model.setParam(gp.GRB.Param.MIPGap, value)
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            self.solver.parameters.relative_gap_limit = value
        else:
            self._print_not_implemented_msg(method="SetMIPGap")

    def SetTimeLimit(self, value: int):
        """Sets the time limit for the optimiser.

        Args:
            value (int): the time limit value. If the value is negative, the behaviour might vary per optimiser software.
                For Gurobi and for CP-SAT, which don't accept negative values, the absolute value will be used.

        Raises:
            UnsupportedOptimiserException: _description_
        """
        if value is None:
            return
        if self.optimiser == OptimiserSoftware.GUROBI:
            self.model.setParam(gp.GRB.Param.TimeLimit, abs(value))
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            self.solver.parameters.max_time_in_seconds = abs(value)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def GetTimeLimit(self) -> int:
        if self.optimiser == OptimiserSoftware.GUROBI:
            return self.model.Params.TimeLimit
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            return self.solver.parameters.max_time_in_seconds
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def SetMIPFocus(self, value: int):
        if self.optimiser == OptimiserSoftware.GUROBI:
            self.model.setParam(gp.GRB.Param.MIPFocus, value)
        else:
            self._print_not_implemented_msg(method="SetMIPFocus")

    def SetHeuristics(self, value: float):
        if self.optimiser == OptimiserSoftware.GUROBI:
            self.model.Params.Heuristics = value
        else:
            self._print_not_implemented_msg(method="SetHeuristics")

    def TurnOffHeuristics(self):
        if self.optimiser == OptimiserSoftware.GUROBI:
            self.model.Params.Heuristics = 0
        else:
            self._print_not_implemented_msg(method="TurnOffHeuristics")

    def WriteVars(self, filename: str, storage):
        vars_dict = self.GetVars()
        data = ""
        for var_name in vars_dict:
            data += "{}\n".format(var_name)
        storage.write(filename, data)

    def WriteModel(self, filename: str):
        if self.optimiser == OptimiserSoftware.GUROBI:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.model.write(filename + ".lp")
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.model.ExportToFile(filename + ".pb.txt")
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def WriteSolution(self, filename: str):
        if self.optimiser == OptimiserSoftware.GUROBI:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.model.write(filename + ".sol")
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            data = ""
            for i, var in enumerate(self.model.Proto().variables):
                v = self.solver.ResponseProto().solution[i]
                if v != 0:
                    data += f"{var.name} = {v}\n"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.write(data)
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def GetSolutionTxt(self) -> str:
        sol = {}
        if self.optimiser == OptimiserSoftware.GUROBI:
            sol = self._GetSolDictGurobi(include_zeros=False)
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            sol = self._GetSolDictORTools(include_zeros=False)
        else:
            raise UnsupportedOptimiserException(self.optimiser)
        data = ""
        for name, value in sol.items():
            data += f"{name} = {value}\n"
        return data

    def SetLogOptimiser(self, value: bool):
        if self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            self.solver.parameters.log_search_progress = value
        else:
            self._print_not_implemented_msg(method="SetLogOptimiser")

    def SetNumThreads(self, value: int):
        if self.optimiser == OptimiserSoftware.GUROBI:
            # Default=0 means 32 threads
            # https://www.gurobi.com/documentation/9.5/refman/threads.html#:~:text=Each%20thread%20can%20consume%20a,parameter%20to%20a%20larger%20value.
            self.model.setParam(gp.GRB.Param.Threads, value)
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            self.solver.parameters.num_search_workers = value
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def SetMipRelStop(self, value: float):
        if self.optimiser == OptimiserSoftware.GUROBI:
            self.model.Params.OptimalityTol = value
        else:
            self._print_not_implemented_msg(method="SetMipRelStop")

    def GetMipRelStop(self) -> float:
        if self.optimiser == OptimiserSoftware.GUROBI:
            return self.model.Params.OptimalityTol
        else:
            v = 0
            self._print_not_implemented_msg(method="GetMipRelStop", value=v)
            return v

    def GetNewCallback(
        self,
        solution_count_limit: Optional[int] = None,
        time_limit_no_improve: Optional[int] = None,
        user_update: Optional[Callable] = None,
    ) -> NewSolGurobiCallback | NewSolCPSatCallBack:
        if self.optimiser == OptimiserSoftware.GUROBI:
            return NewSolGurobiCallback(
                time_limit=self.GetTimeLimit(),
                model=self.model,
                solution_count_limit=solution_count_limit,
                time_limit_no_improve=time_limit_no_improve,
                user_update=user_update,
            )
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            return NewSolCPSatCallBack(
                time_limit=self.GetTimeLimit(),
                solution_count_limit=solution_count_limit,
                time_limit_no_improve=time_limit_no_improve,
            )
        raise UnsupportedOptimiserException(self.optimiser)

    def Optimise(
        self,
        sol_callback: Union[NewSolGurobiCallback, NewSolCPSatCallBack, None] = None,
    ) -> Any:
        logging.info("Solving...")
        if self.optimiser == OptimiserSoftware.GUROBI:
            if not sol_callback:
                sol_callback = self.GetNewCallback()
            assert isinstance(sol_callback, NewSolGurobiCallback)
            self.sol_callback = sol_callback
            self.sol_callback.timer._reset_timer()
            self.sol_callback.timer._reset_total_timer()
            self.model.optimize(sol_callback.callback)
            self.solver_status = self.model.Status
            self.sol_callback.timer.cancel_timer()
            self.sol_callback.timer.cancel_total_timer()
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            if not sol_callback:
                sol_callback = self.GetNewCallback()
            assert isinstance(sol_callback, NewSolCPSatCallBack)
            self.sol_callback = sol_callback
            self.sol_callback._reset_timer()
            self.sol_callback._reset_total_timer()
            self.solver_status = self.solver.Solve(self.model, sol_callback)
            self.sol_callback.cancel_timer()
            self.sol_callback.cancel_total_timer()
        else:
            raise UnsupportedOptimiserException(self.optimiser)
        logging.info(f"Solver status = {self.solver_status}")
        return self.solver_status

    def GetStatus(self) -> OptStatus:
        if self.optimiser == OptimiserSoftware.GUROBI:
            if self.model.SolCount >= 1:
                if self.solver_status == gp.GRB.OPTIMAL:
                    return OptStatus.OPTIMAL
                return OptStatus.FEASIBLE
            if self.solver_status == gp.GRB.INFEASIBLE:
                return OptStatus.INFEASIBLE
            return OptStatus.NO_SOLUTION_FOUND
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            if self.solver_status == cp_model.OPTIMAL:
                return OptStatus.OPTIMAL
            elif self.solver_status == cp_model.FEASIBLE:
                return OptStatus.FEASIBLE
            elif self.solver_status == cp_model.INFEASIBLE:
                return OptStatus.INFEASIBLE
            return OptStatus.NO_SOLUTION_FOUND
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def GetGap(self) -> Union[float, None]:
        if (
            self.GetStatus() != OptStatus.FEASIBLE
            and self.GetStatus() != OptStatus.OPTIMAL
        ):
            return None
        obj = self.GetObjValue()
        bestbound = self.GetBestBound()
        gap = 100 * (obj - bestbound) / (obj if obj != 0 else 1)
        return gap

    def Succeeded(self) -> bool:
        status = self.GetStatus()
        return status == OptStatus.OPTIMAL or status == OptStatus.FEASIBLE

    def GetNumSols(self) -> int:
        if self.optimiser == OptimiserSoftware.GUROBI:
            return self.model.SolCount
        if self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            assert self.sol_callback is not None
            return self.sol_callback.solution_count
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def GetTime1stSol(self) -> Union[int, None]:
        if self.sol_callback is None:
            print("Warning: not callback available to retrieve time of 1st solution")
            return None
        if (
            self.optimiser == OptimiserSoftware.GUROBI
            or self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT
        ):
            return self.sol_callback.time_1st_sol
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def CalculateIIS(self, filename: str):
        if self.optimiser == OptimiserSoftware.GUROBI:
            if self.solver_status == gp.GRB.INFEASIBLE:
                self.model.computeIIS()
                self.model.write(filename + ".ilp")
            return
        self._print_not_implemented_msg(method="CalculateIIS")

    def GetOptimiserRunTime(self) -> float:
        if self.optimiser == OptimiserSoftware.GUROBI:
            return self.model.runtime
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            return float(self.solver.WallTime())
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def PrintModelStats(self):
        if self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            logging.debug(f"Model stats = {self.model.ModelStats()}")
            return
        self._print_not_implemented_msg(method="PrintModelStats")

    def DisposeModel(self):
        if self.optimiser == OptimiserSoftware.GUROBI:
            self.model.dispose()
        elif self.optimiser == OptimiserSoftware.ORTOOLS_CPSAT:
            self._print_not_implemented_msg(method="DisposeModel", warning=False)
            return
        else:
            raise UnsupportedOptimiserException(self.optimiser)

    def _print_not_implemented_msg(
        self,
        method: str,
        value: Any = None,
        extra_msg: Optional[str] = None,
        warning=True,
    ):
        prefix = "Warning" if warning else "Info"
        msg = f"{prefix}: method {method} not implement for opt software {self.optimiser}. Returned value = {value}. "
        if extra_msg:
            msg += extra_msg
        if warning:
            logging.warning(msg)
        else:
            logging.info(msg)


class UnsupportedOptimiserException(Exception):
    def __init__(self, optimiser: OptimiserSoftware):
        name: str = str(optimiser)
        if type(optimiser) is OptimiserSoftware:
            name = optimiser.name
        self.message = f"The optimiser {name} is not supported by Optimiser or by the used function."
        super().__init__(self.message)
