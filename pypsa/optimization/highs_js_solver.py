# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""HiGHS JavaScript/WebAssembly solver bridge for Pyodide."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linopy import Model

logger = logging.getLogger(__name__)


def solve_with_highs_js(model: Model, **kwargs) -> tuple[str, str]:
    """Solve optimization model using HiGHS JavaScript/WebAssembly solver.

    This function extracts the linear programming problem from the linopy model,
    converts it to HiGHS JSON format, calls the JavaScript HiGHS solver,
    and converts the solution back.

    Parameters
    ----------
    model : linopy.Model
        The optimization model to solve
    **kwargs
        Additional solver options (currently ignored)

    Returns
    -------
    status : str
        Solver status ("ok" or error status)
    condition : str
        Termination condition ("optimal", "infeasible", etc.)
    """
    try:
        import js
        import numpy as np
        import scipy.sparse as sp
    except ImportError as e:
        msg = f"HiGHS-JS solver requires Pyodide environment: {e}"
        logger.error(msg)
        return "error", "unknown"

    logger.info("Extracting model for HiGHS-JS solver...")

    # Get constraint matrix and bounds
    A, b_lower, b_upper, c, sense = _extract_model_data(model)

    # Get variable bounds
    v_lower, v_upper = _extract_variable_bounds(model)

    # Build HiGHS problem dictionary
    problem = _build_highs_problem(A, b_lower, b_upper, c, v_lower, v_upper, sense)

    logger.info(f"Calling HiGHS-JS solver with {len(problem['cols'])} variables and {len(problem['rows'])} constraints...")

    # Call JavaScript HiGHS solver
    try:
        result = js.js_highs_solve(problem)
    except Exception as e:
        msg = f"HiGHS-JS solver failed: {e}"
        logger.error(msg)
        return "error", "unknown"

    # Parse results
    status, condition = _parse_highs_result(result, model)

    return status, condition


def _extract_model_data(model: Model):
    """Extract constraint matrix and objective from linopy model.

    Uses linopy's built-in matrix accessor for proper model conversion.
    """
    import numpy as np
    import scipy.sparse as sp

    # Use linopy's matrix accessor to get properly formatted matrices
    matrices = model.matrices

    # Get constraint matrix A (rows=constraints, cols=variables)
    A = matrices.A
    if A is None:
        # No constraints - create empty matrix
        n_vars = len(matrices.vlabels)
        A = sp.csr_matrix((0, n_vars))

    # Get objective coefficients
    c = matrices.c
    if c is None:
        c = np.zeros(len(matrices.vlabels))
    else:
        c = c.values

    # Get constraint bounds
    b = matrices.b
    sense = matrices.sense

    # Convert constraint bounds to lower/upper format
    b_lower = np.full(len(b), -np.inf)
    b_upper = np.full(len(b), np.inf)

    for i, (b_val, s) in enumerate(zip(b, sense)):
        if s == "==":
            b_lower[i] = b_val
            b_upper[i] = b_val
        elif s == ">=":
            b_lower[i] = b_val
            b_upper[i] = np.inf
        elif s == "<=":
            b_lower[i] = -np.inf
            b_upper[i] = b_val

    # Get objective sense
    objective_sense = "minimize" if model.objective.sense == "min" else "maximize"

    return A, b_lower, b_upper, c, objective_sense


def _extract_variable_bounds(model: Model):
    """Extract variable bounds from linopy model.

    Uses linopy's built-in matrix accessor for proper bounds extraction.
    """
    import numpy as np

    matrices = model.matrices

    # Get variable bounds from matrices accessor
    v_lower = matrices.lb.values
    v_upper = matrices.ub.values

    # Replace NaN with infinity
    v_lower = np.where(np.isnan(v_lower), -np.inf, v_lower)
    v_upper = np.where(np.isnan(v_upper), np.inf, v_upper)

    return v_lower, v_upper


def _build_highs_problem(A, b_lower, b_upper, c, v_lower, v_upper, sense):
    """Build HiGHS JSON problem format."""
    import numpy as np

    problem = {
        "sense": sense,
        "offset": 0.0,
        "cols": [],
        "rows": [],
    }

    # Add variables (columns)
    for i in range(len(c)):
        col = {
            "name": f"x{i}",
            "obj": float(c[i]) if not np.isnan(c[i]) else 0.0,
        }
        if not np.isinf(v_lower[i]):
            col["lb"] = float(v_lower[i])
        if not np.isinf(v_upper[i]):
            col["ub"] = float(v_upper[i])
        problem["cols"].append(col)

    # Add constraints (rows)
    for i in range(A.shape[0]):
        row = {"coeffs": []}
        row_data = A.getrow(i)
        for j, val in zip(row_data.indices, row_data.data):
            row["coeffs"].append({"col": int(j), "val": float(val)})

        if not np.isinf(b_lower[i]):
            row["lb"] = float(b_lower[i])
        if not np.isinf(b_upper[i]):
            row["ub"] = float(b_upper[i])

        problem["rows"].append(row)

    return problem


def _parse_highs_result(result, model: Model):
    """Parse HiGHS result and update model solution."""
    import numpy as np
    import xarray as xr

    # Check status
    status_map = {
        "Optimal": ("ok", "optimal"),
        "Infeasible": ("warning", "infeasible"),
        "Unbounded": ("warning", "unbounded"),
    }

    result_status = result.get("Status", "Unknown")
    status, condition = status_map.get(result_status, ("error", "unknown"))

    if status != "ok":
        logger.warning(f"HiGHS-JS solver finished with status: {result_status}")
        return status, condition

    # Extract solution values
    solution_values = np.array([col.get("Primal", 0.0) for col in result.get("Columns", [])])

    # Assign solution to model variables
    idx = 0
    for var_name in model.variables:
        var = model.variables[var_name]
        size = var.size
        var_solution = solution_values[idx:idx+size].reshape(var.shape)

        # Create xarray with proper coordinates
        var.solution = xr.DataArray(var_solution, coords=var.coords, dims=var.dims)

        idx += size

    # Set objective value if available
    if "ObjectiveValue" in result:
        model.objective.value = result["ObjectiveValue"]

    logger.info(f"HiGHS-JS solver completed successfully. Objective: {model.objective.value}")

    return status, condition
