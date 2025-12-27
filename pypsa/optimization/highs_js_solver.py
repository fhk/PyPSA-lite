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
        # The JS function is async, so we get a PyodideFuture
        # We need to await it to get the actual result
        import asyncio
        result_future = js.js_highs_solve(lp_string)

        # Convert PyodideFuture to actual result
        # In Pyodide, we can use asyncio.ensure_future and run it
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(result_future)
    except Exception as e:
        msg = f"HiGHS-JS solver failed: {e}"
        logger.error(msg)
        return "error", "unknown"

    # Parse results
    status, condition = _parse_highs_result(result, model)

    return status, condition


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
