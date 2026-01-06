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

    This function exports the linopy model to LP format, calls the JavaScript
    HiGHS solver via Pyodide's bridge, and parses the solution back into the model.

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
    except ImportError as e:
        msg = f"HiGHS-JS solver requires Pyodide environment: {e}"
        logger.error(msg)
        return "error", "unknown"

    logger.info("Converting model to LP format for HiGHS-JS solver...")

    # Export model to LP format via Pyodide's virtual filesystem
    from pathlib import Path

    # Use a simple path in Pyodide's virtual filesystem
    lp_path = Path("/tmp/highs_problem.lp")

    # Write model to LP file
    model.to_file(lp_path, io_api="lp")

    # Read LP file as string
    lp_string = lp_path.read_text()

    # Clean up
    lp_path.unlink()

    logger.info(f"Calling HiGHS-JS solver...")

    # Call JavaScript HiGHS solver (async function)
    try:
        import time

        # Reset the global state before calling
        js.globalThis.highs_solve_pending = False
        js.globalThis.highs_last_result = None

        # Call the async function - it will store result in globalThis.highs_last_result
        logger.info("Calling HiGHS-JS solver...")
        js.js_highs_solve(lp_string)

        # Poll until solve is complete
        timeout = 30  # seconds
        start_time = time.time()

        logger.info("Waiting for HiGHS solver to complete...")

        # Wait for either: pending to become true (solver started), or result to be available
        while (time.time() - start_time) < timeout:
            time.sleep(0.05)  # Let the browser event loop run

            # If we have a result, we're done
            if js.highs_last_result is not None:
                break

            # If solver is running (pending=True), continue waiting
            if js.highs_solve_pending:
                continue

            # If pending is False and no result yet, keep waiting (solver might not have started yet)

        # Check if we timed out
        if js.highs_last_result is None:
            if js.highs_solve_pending:
                raise Exception("Timeout waiting for HiGHS solver (still running)")
            else:
                raise Exception("Timeout waiting for HiGHS solver (never started or no result)")

        # Get the result from the global variable
        result = js.highs_last_result

        print(f"[DEBUG] HiGHS-JS result received, type: {type(result)}")
        logger.info(f"HiGHS-JS result received")

        # Convert JsProxy to Python dict using to_py()
        if hasattr(result, 'to_py'):
            result = result.to_py()
            print(f"[DEBUG] Converted result to Python dict. Keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            logger.info(f"Converted result to Python dict")
        else:
            print(f"[DEBUG] ERROR: Result doesn't have to_py() method")
            logger.error("Result doesn't have to_py() method")
            return "error", "unknown"

    except Exception as e:
        msg = f"HiGHS-JS solver failed: {e}"
        logger.error(msg)
        return "error", "unknown"

    # Parse results
    status, condition = _parse_highs_result(result, model)

    return status, condition


def _parse_highs_result(result, model: Model):
    """Parse HiGHS result and update model solution.

    Parameters
    ----------
    result : dict
        Python dict converted from HiGHS-JS result using to_py()
    model : linopy.Model
        The optimization model to update with solution
    """
    import numpy as np
    import xarray as xr

    # Check status (result is now a Python dict)
    status_map = {
        "Optimal": ("ok", "optimal"),
        "Infeasible": ("warning", "infeasible"),
        "Unbounded": ("warning", "unbounded"),
    }

    result_status = result.get("Status", "Unknown")
    status, condition = status_map.get(result_status, ("error", "unknown"))

    if status != "ok":
        print(f"[DEBUG] HiGHS-JS solver finished with status: {result_status}")
        print(f"[DEBUG] Available keys in result: {list(result.keys())}")
        print(f"[DEBUG] Full result: {result}")
        logger.warning(f"HiGHS-JS solver finished with status: {result_status}")
        logger.warning(f"Available status values in result: {list(result.keys())}")
        logger.warning(f"Full result: {result}")
        return status, condition

    # Extract solution values from Columns dict
    columns = result.get("Columns", {})

    # Sort by variable name (x0, x1, x2, ...) to ensure correct order
    sorted_vars = sorted(columns.keys(), key=lambda x: int(x[1:]) if x.startswith('x') else x)

    # Extract Primal values
    solution_values = np.array([columns[var_name].get("Primal", 0.0) for var_name in sorted_vars])

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
    objective_value = result.get("ObjectiveValue")
    if objective_value is not None:
        # Set internal _value attribute (same as linopy's solve method does)
        model.objective._value = objective_value
    else:
        logger.warning("ObjectiveValue not found in HiGHS result")

    logger.info(f"HiGHS-JS solver completed successfully. Objective: {model.objective.value}")

    # Mark model as optimized by setting status
    model.status = status
    model.termination_condition = condition

    return status, condition
