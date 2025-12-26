"""Test suite for index.html HiGHS-JS bridge and browser integration.

This test suite uses Playwright to test the browser-based components:
- HiGHS-JS loader initialization
- Pyodide initialization
- PyPSA installation
- Python-JavaScript bridge
- UI interactions
- End-to-end optimization workflow

Requirements:
    pip install pytest pytest-playwright
    playwright install chromium
"""

import pytest
import asyncio
import re
from playwright.async_api import async_playwright, Page, expect
from pathlib import Path


# Configuration
INDEX_HTML_PATH = Path(__file__).parent / "index.html"
TIMEOUT_SHORT = 10000  # 10 seconds
TIMEOUT_MEDIUM = 30000  # 30 seconds
TIMEOUT_LONG = 120000  # 2 minutes for full initialization


@pytest.fixture
async def page():
    """Fixture to provide a browser page for testing."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Enable console logging for debugging
        page.on("console", lambda msg: print(f"[Browser Console] {msg.type}: {msg.text}"))
        page.on("pageerror", lambda exc: print(f"[Browser Error] {exc}"))

        yield page

        await context.close()
        await browser.close()


@pytest.fixture
async def loaded_page(page: Page):
    """Fixture that provides a page with index.html loaded."""
    await page.goto(f"file://{INDEX_HTML_PATH.absolute()}")
    await page.wait_for_load_state("networkidle")
    yield page


class TestScriptLoading:
    """Test that required scripts load correctly."""

    @pytest.mark.asyncio
    async def test_pyodide_script_loads(self, loaded_page: Page):
        """Test that Pyodide script tag is present and loads."""
        # Check script tag exists
        pyodide_script = await loaded_page.query_selector('script[src*="pyodide"]')
        assert pyodide_script is not None, "Pyodide script tag not found"

        # Verify loadPyodide function is available
        is_available = await loaded_page.evaluate("typeof loadPyodide === 'function'")
        assert is_available, "loadPyodide function not available"

    @pytest.mark.asyncio
    async def test_highs_script_loads(self, loaded_page: Page):
        """Test that HiGHS-JS script tag is present and loads."""
        # Check script tag exists
        highs_script = await loaded_page.query_selector('script[src*="highs-js"]')
        assert highs_script is not None, "HiGHS-JS script tag not found"

        # Verify highs loader is available
        is_available = await loaded_page.evaluate("typeof highs === 'function'")
        assert is_available, "highs loader function not available"


class TestHiGHSInitialization:
    """Test HiGHS-JS loader initialization."""

    @pytest.mark.asyncio
    async def test_highs_loader_with_locatefile(self, loaded_page: Page):
        """Test that HiGHS loader initializes with locateFile correctly."""
        result = await loaded_page.evaluate("""
            async () => {
                try {
                    const highs_loader = highs;
                    const solver = await highs_loader({
                        locateFile: (file) => "https://lovasoa.github.io/highs-js/" + file
                    });

                    return {
                        success: true,
                        hasSolveMethod: typeof solver.solve === 'function',
                        hasOtherMethods: typeof solver.Highs === 'function'
                    };
                } catch (error) {
                    return {
                        success: false,
                        error: error.message
                    };
                }
            }
        """)

        assert result["success"], f"HiGHS initialization failed: {result.get('error')}"
        assert result["hasSolveMethod"], "HiGHS solver missing solve method"

    @pytest.mark.asyncio
    async def test_solve_with_highs_function_exists(self, loaded_page: Page):
        """Test that solveWithHiGHS bridge function exists."""
        exists = await loaded_page.evaluate("typeof solveWithHiGHS === 'function'")
        assert exists, "solveWithHiGHS function not defined"

    @pytest.mark.asyncio
    async def test_solve_with_highs_initializes_solver(self, loaded_page: Page):
        """Test that solveWithHiGHS initializes the solver lazily."""
        result = await loaded_page.evaluate("""
            async () => {
                // Create a simple LP problem
                const problem = {
                    sense: 'minimize',
                    cols: [
                        { obj: 1.0, lb: 0.0 },
                        { obj: 2.0, lb: 0.0 }
                    ],
                    rows: [
                        { lb: 1.0, coeffs: [0, 1], vars: [0, 1] }
                    ]
                };

                try {
                    const result = await solveWithHiGHS(problem);
                    return {
                        success: true,
                        hasStatus: 'Status' in result || 'status' in result
                    };
                } catch (error) {
                    return {
                        success: false,
                        error: error.message
                    };
                }
            }
        """, timeout=TIMEOUT_MEDIUM)

        assert result["success"], f"solveWithHiGHS failed: {result.get('error')}"
        assert result["hasStatus"], "Solver result missing status field"


class TestPyodideInitialization:
    """Test Pyodide initialization and setup."""

    @pytest.mark.asyncio
    async def test_pyodide_loads(self, loaded_page: Page):
        """Test that Pyodide initializes successfully."""
        # Wait for "Pyodide loaded!" message in output
        await loaded_page.wait_for_function(
            """() => {
                const output = document.getElementById('output');
                return output && output.value.includes('Pyodide loaded!');
            }""",
            timeout=TIMEOUT_LONG
        )

        output_text = await loaded_page.input_value("#output")
        assert "Pyodide loaded!" in output_text

    @pytest.mark.asyncio
    async def test_highs_solver_loads_in_main(self, loaded_page: Page):
        """Test that HiGHS solver loads during initialization."""
        await loaded_page.wait_for_function(
            """() => {
                const output = document.getElementById('output');
                return output && output.value.includes('HiGHS solver loaded!');
            }""",
            timeout=TIMEOUT_LONG
        )

        output_text = await loaded_page.input_value("#output")
        assert "HiGHS solver loaded!" in output_text

    @pytest.mark.asyncio
    async def test_micropip_loads(self, loaded_page: Page):
        """Test that micropip is loaded."""
        await loaded_page.wait_for_function(
            """() => {
                const output = document.getElementById('output');
                return output && output.value.includes('Loading micropip');
            }""",
            timeout=TIMEOUT_LONG
        )

        output_text = await loaded_page.input_value("#output")
        assert "Loading micropip" in output_text


class TestPyPSAIntegration:
    """Test PyPSA installation and integration."""

    @pytest.mark.asyncio
    async def test_pypsa_installs(self, loaded_page: Page):
        """Test that PyPSA installs from GitHub Pages."""
        await loaded_page.wait_for_function(
            """() => {
                const output = document.getElementById('output');
                return output && output.value.includes('PyPSA installed successfully!');
            }""",
            timeout=TIMEOUT_LONG
        )

        output_text = await loaded_page.input_value("#output")
        assert "PyPSA installed successfully!" in output_text

    @pytest.mark.asyncio
    async def test_highs_js_bridge_ready(self, loaded_page: Page):
        """Test that HiGHS-JS solver bridge is ready."""
        await loaded_page.wait_for_function(
            """() => {
                const output = document.getElementById('output');
                return output && output.value.includes('HiGHS-JS solver bridge is ready!');
            }""",
            timeout=TIMEOUT_LONG
        )

        output_text = await loaded_page.input_value("#output")
        assert "HiGHS-JS solver bridge is ready!" in output_text

    @pytest.mark.asyncio
    async def test_js_highs_solve_exposed_to_python(self, loaded_page: Page):
        """Test that js_highs_solve is available in Python via pyodide.globals."""
        # Wait for initialization to complete
        await loaded_page.wait_for_function(
            """() => {
                const output = document.getElementById('output');
                return output && output.value.includes('Ready!');
            }""",
            timeout=TIMEOUT_LONG
        )

        # Check if the bridge function was set in pyodide.globals
        has_bridge = await loaded_page.evaluate("""
            async () => {
                const pyodide = await pyodideReadyPromise;
                return pyodide.globals.has('js_highs_solve');
            }
        """)

        assert has_bridge, "js_highs_solve not exposed to Python"


class TestUIInteraction:
    """Test UI interactions and evaluatePython function."""

    @pytest.mark.asyncio
    async def test_ui_elements_exist(self, loaded_page: Page):
        """Test that all UI elements are present."""
        # Wait for page to be ready
        await loaded_page.wait_for_function(
            """() => {
                const output = document.getElementById('output');
                return output && output.value.includes('Ready!');
            }""",
            timeout=TIMEOUT_LONG
        )

        # Check elements exist
        code_input = await loaded_page.query_selector("#code")
        assert code_input is not None, "Code input not found"

        run_button = await loaded_page.query_selector("button")
        assert run_button is not None, "Run button not found"

        output_textarea = await loaded_page.query_selector("#output")
        assert output_textarea is not None, "Output textarea not found"

    @pytest.mark.asyncio
    async def test_evaluate_python_simple(self, loaded_page: Page):
        """Test evaluating simple Python code."""
        # Wait for initialization
        await loaded_page.wait_for_function(
            """() => {
                const output = document.getElementById('output');
                return output && output.value.includes('Ready!');
            }""",
            timeout=TIMEOUT_LONG
        )

        # Clear input and enter test code
        await loaded_page.fill("#code", "2 + 2")

        # Get initial output length
        initial_output = await loaded_page.input_value("#output")

        # Click run button
        await loaded_page.click("button")

        # Wait for output to update
        await asyncio.sleep(1)

        # Check output contains result
        output_text = await loaded_page.input_value("#output")
        assert ">>> 2 + 2" in output_text
        assert "4" in output_text

    @pytest.mark.asyncio
    async def test_evaluate_python_with_numpy(self, loaded_page: Page):
        """Test evaluating Python code that imports numpy."""
        # Wait for initialization
        await loaded_page.wait_for_function(
            """() => {
                const output = document.getElementById('output');
                return output && output.value.includes('Ready!');
            }""",
            timeout=TIMEOUT_LONG
        )

        # Test numpy import
        await loaded_page.fill("#code", "import numpy as np; np.array([1, 2, 3]).sum()")
        await loaded_page.click("button")

        # Wait a bit for execution
        await asyncio.sleep(2)

        output_text = await loaded_page.input_value("#output")
        assert "6" in output_text or "array" in output_text


class TestDemoExecution:
    """Test the demo execution that runs on page load."""

    @pytest.mark.asyncio
    async def test_demo_runs_successfully(self, loaded_page: Page):
        """Test that the demo optimization runs successfully."""
        # Wait for demo to complete
        await loaded_page.wait_for_function(
            """() => {
                const output = document.getElementById('output');
                return output && output.value.includes('Running demo...');
            }""",
            timeout=TIMEOUT_LONG
        )

        output_text = await loaded_page.input_value("#output")
        assert "Running demo..." in output_text

    @pytest.mark.asyncio
    async def test_demo_creates_network(self, loaded_page: Page):
        """Test that demo creates PyPSA network."""
        # Wait for ready state
        await loaded_page.wait_for_function(
            """() => {
                const output = document.getElementById('output');
                return output && output.value.includes('Ready!');
            }""",
            timeout=TIMEOUT_LONG
        )

        output_text = await loaded_page.input_value("#output")
        # The demo should print some output about the network
        assert "Running demo..." in output_text

    @pytest.mark.asyncio
    async def test_page_ready_state(self, loaded_page: Page):
        """Test that page reaches ready state."""
        await loaded_page.wait_for_function(
            """() => {
                const output = document.getElementById('output');
                return output && output.value.includes('Ready! You can now run Python code.');
            }""",
            timeout=TIMEOUT_LONG
        )

        output_text = await loaded_page.input_value("#output")
        assert "Ready! You can now run Python code." in output_text


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, loaded_page: Page):
        """Test complete workflow from page load to Python execution."""
        # 1. Wait for all initialization steps
        await loaded_page.wait_for_function(
            """() => {
                const output = document.getElementById('output');
                return output && output.value.includes('Ready!');
            }""",
            timeout=TIMEOUT_LONG
        )

        output_text = await loaded_page.input_value("#output")

        # 2. Verify all initialization steps completed
        assert "Pyodide loaded!" in output_text
        assert "HiGHS solver loaded!" in output_text
        assert "PyPSA installed successfully!" in output_text
        assert "HiGHS-JS solver bridge is ready!" in output_text

        # 3. Execute custom Python code
        await loaded_page.fill("#code", "print('Test passed!')")
        await loaded_page.click("button")

        await asyncio.sleep(2)

        # 4. Verify execution worked
        final_output = await loaded_page.input_value("#output")
        assert "Test passed!" in final_output

    @pytest.mark.asyncio
    async def test_pypsa_network_creation_via_ui(self, loaded_page: Page):
        """Test creating a PyPSA network via UI."""
        # Wait for ready
        await loaded_page.wait_for_function(
            """() => {
                const output = document.getElementById('output');
                return output && output.value.includes('Ready!');
            }""",
            timeout=TIMEOUT_LONG
        )

        # Create a simple network
        code = """
import pypsa
n = pypsa.Network()
n.add("Bus", "test_bus")
print(f"Created network with {len(n.buses)} bus(es)")
        """.strip()

        await loaded_page.fill("#code", code)
        await loaded_page.click("button")

        await asyncio.sleep(2)

        output_text = await loaded_page.input_value("#output")
        assert "Created network with 1 bus(es)" in output_text


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_python_syntax_error(self, loaded_page: Page):
        """Test that Python syntax errors are handled gracefully."""
        # Wait for ready
        await loaded_page.wait_for_function(
            """() => {
                const output = document.getElementById('output');
                return output && output.value.includes('Ready!');
            }""",
            timeout=TIMEOUT_LONG
        )

        # Enter invalid Python code
        await loaded_page.fill("#code", "this is not valid python !!!!")
        await loaded_page.click("button")

        await asyncio.sleep(1)

        output_text = await loaded_page.input_value("#output")
        # Should show some error
        assert "Error" in output_text or "Syntax" in output_text or "invalid" in output_text

    @pytest.mark.asyncio
    async def test_python_runtime_error(self, loaded_page: Page):
        """Test that Python runtime errors are handled gracefully."""
        # Wait for ready
        await loaded_page.wait_for_function(
            """() => {
                const output = document.getElementById('output');
                return output && output.value.includes('Ready!');
            }""",
            timeout=TIMEOUT_LONG
        )

        # Enter code that will raise an error
        await loaded_page.fill("#code", "1 / 0")
        await loaded_page.click("button")

        await asyncio.sleep(1)

        output_text = await loaded_page.input_value("#output")
        assert "Error" in output_text or "Division" in output_text or "ZeroDivision" in output_text


if __name__ == "__main__":
    # Run with: python -m pytest test_highs_js_bridge.py -v
    print("Run this test suite with: python -m pytest test_highs_js_bridge.py -v")
    print("Make sure to install dependencies first:")
    print("  pip install pytest pytest-playwright")
    print("  playwright install chromium")
