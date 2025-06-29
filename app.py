import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import time
import os
import json
import hashlib
import pathlib
from PIL import Image # Added for loading images from cache
import os # Import os module

# Disable Hugging Face Hub telemetry for local development
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# --- Caching Setup ---
CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def _generate_cache_key(*args, **kwargs):
    """Generates a unique hash for the given arguments."""
    # Convert all arguments to a consistent string representation
    # Exclude the 'progress' object from hashing as it's not part of the configuration
    hash_input = []
    for arg in args:
        if not isinstance(arg, gr.Progress):
            hash_input.append(str(arg))
    for k, v in kwargs.items():
        if k != 'progress':
            hash_input.append(f"{k}={v}")
    
    # Use a stable JSON representation for dictionary arguments if any
    # For this specific function, all args are simple types, so direct string conversion is fine.
    
    return hashlib.md5("".join(hash_input).encode('utf-8')).hexdigest()

def _save_to_cache(key, results_text, fig1, fig2):
    """Saves simulation results and plots to the cache."""
    cache_path = CACHE_DIR / key
    cache_path.mkdir(exist_ok=True)

    # Save plots as PNGs
    fig1_path = cache_path / "fig1.png"
    fig2_path = cache_path / "fig2.png"
    fig1.savefig(fig1_path)
    fig2.savefig(fig2_path)
    plt.close(fig1) # Close figures to free memory
    plt.close(fig2)

    # Save metadata (results_text and plot paths)
    metadata = {
        "results_text": results_text,
        "fig1_path": str(fig1_path),
        "fig2_path": str(fig2_path)
    }
    with open(cache_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

def _load_from_cache(key):
    """Loads simulation results from the cache."""
    cache_path = CACHE_DIR / key
    metadata_path = cache_path / "metadata.json"

    if not metadata_path.exists():
        return None

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Load images from paths
    try:
        fig1_img = Image.open(metadata["fig1_path"])
        fig2_img = Image.open(metadata["fig2_path"])
    except FileNotFoundError:
        print(f"Cached image files not found for key: {key}. Deleting cache entry.")
        # Clean up incomplete cache entry
        import shutil
        shutil.rmtree(cache_path)
        return None

    # Convert PIL Image objects back to matplotlib figures for Gradio's gr.Plot
    # This is a workaround as gr.Plot expects matplotlib figures, not PIL Images directly.
    fig1_recreated = plt.figure()
    ax1_recreated = fig1_recreated.add_subplot(111)
    ax1_recreated.imshow(fig1_img)
    ax1_recreated.axis('off') # Hide axes for image display

    fig2_recreated = plt.figure()
    ax2_recreated = fig2_recreated.add_subplot(111)
    ax2_recreated.imshow(fig2_img)
    ax2_recreated.axis('off') # Hide axes for image display

    return "Loaded from cache!", metadata["results_text"], fig1_recreated, fig2_recreated

def run_simulation(
    initial_investment: float,
    num_years: int,
    num_simulations: int,
    target_success_rate: float,
    stock_mean_return: float,
    stock_std_dev: float,
    bond_mean_return: float,
    bond_std_dev: float,
    stock_allocation: float,
    correlation_stock_bond: float,
    mean_inflation: float,
    std_dev_inflation: float,
    min_swr_test: float,
    max_swr_test: float,
    num_swr_intervals: int,
    progress=gr.Progress()
):
    # Generate cache key from all input parameters except 'progress'
    cache_key = _generate_cache_key(
        initial_investment, num_years, num_simulations, target_success_rate,
        stock_mean_return, stock_std_dev, bond_mean_return, bond_std_dev,
        stock_allocation, correlation_stock_bond, mean_inflation,
        std_dev_inflation, min_swr_test, max_swr_test, num_swr_intervals
    )

    # Check if results are in cache
    cached_results = _load_from_cache(cache_key)
    if cached_results:
        progress(1, desc="Loading from cache...")
        return cached_results

    swr_test_step = (max_swr_test - min_swr_test) / num_swr_intervals if num_swr_intervals > 0 else 0.1 # Calculate step
    progress(0, desc="Starting simulation...")
    start_time = time.time()
    # --- Core Parameters ---
    initial_investment = float(initial_investment)
    num_years = int(num_years)
    num_simulations = int(num_simulations)
    target_success_rate = float(target_success_rate) / 100.0 # Convert percentage to decimal

    # --- Financial Assumptions (NOMINAL) ---
    stock_mean_return = float(stock_mean_return) / 100.0
    stock_std_dev = float(stock_std_dev) / 100.0
    bond_mean_return = float(bond_mean_return) / 100.0
    bond_std_dev = float(bond_std_dev) / 100.0
    stock_allocation = float(stock_allocation) / 100.0
    bond_allocation = 1.0 - stock_allocation
    correlation_stock_bond = float(correlation_stock_bond)

    mean_inflation = float(mean_inflation) / 100.0
    std_dev_inflation = float(std_dev_inflation) / 100.0

    # --- Covariance Matrix for generating correlated asset returns ---
    mean_asset_returns = np.array([stock_mean_return, bond_mean_return])
    cov_matrix = np.array([
        [stock_std_dev**2, correlation_stock_bond * stock_std_dev * bond_std_dev],
        [correlation_stock_bond * stock_std_dev * bond_std_dev, bond_std_dev**2]
    ])

    # --- SWRs to Test ---
    # Calculate swr_test_step based on range and number of intervals
    swr_test_step_calculated = (max_swr_test - min_swr_test) / num_swr_intervals if num_swr_intervals > 0 else 0.1
    # Use linspace for more precise interval generation
    withdrawal_rates_to_test = np.linspace(min_swr_test / 100.0, max_swr_test / 100.0, num_swr_intervals + 1)
    all_results = []
    portfolio_paths_for_plotting = {}

    total_swr_tests = len(withdrawal_rates_to_test)
    for idx, swr in enumerate(withdrawal_rates_to_test):
        elapsed_time = time.time() - start_time
        progress_ratio = (idx + 1) / total_swr_tests
        
        if progress_ratio > 0:
            estimated_total_time = elapsed_time / progress_ratio
            estimated_remaining_time = estimated_total_time - elapsed_time
            remaining_minutes = estimated_remaining_time / 60
            progress_desc = f"Simulating SWR: {swr*100:.1f}% (Est. remaining: {remaining_minutes:.1f} min)"
        else:
            progress_desc = f"Simulating SWR: {swr*100:.1f}%"
        
        progress((idx + 1) / total_swr_tests, desc=progress_desc)
        
        success_count = 0
        current_swr_paths = []

        for i in range(num_simulations):
            portfolio_value = float(initial_investment)
            current_annual_withdrawal_nominal = initial_investment * swr
            
            simulation_failed_this_run = False
            path = [portfolio_value]

            for year in range(num_years):
                if year > 0:
                    yearly_inflation = np.random.normal(mean_inflation, std_dev_inflation)
                    yearly_inflation = max(yearly_inflation, -0.05)
                    current_annual_withdrawal_nominal *= (1 + yearly_inflation)

                portfolio_value -= current_annual_withdrawal_nominal

                if portfolio_value <= 0:
                    simulation_failed_this_run = True
                    portfolio_value = 0
                    path.append(portfolio_value)
                    break

                asset_returns_this_year = np.random.multivariate_normal(mean_asset_returns, cov_matrix)
                portfolio_return_this_year = (stock_allocation * asset_returns_this_year[0] +
                                              bond_allocation * asset_returns_this_year[1])
                portfolio_return_this_year = max(portfolio_return_this_year, -0.99)

                portfolio_value *= (1 + portfolio_return_this_year)
                path.append(portfolio_value)

            if not simulation_failed_this_run:
                success_count += 1
            
            # Always append path, we'll select a sample later
            current_swr_paths.append(path)

        success_probability = success_count / num_simulations
        all_results.append({'swr': swr, 'success_rate': success_probability})
        
        # Store a sample of paths for this SWR
        portfolio_paths_for_plotting[swr] = current_swr_paths[:100] # Store up to 100 paths

    final_swr = 0.0
    initial_annual_withdrawal_amount = 0.0

    eligible_rates = [r for r in all_results if r['success_rate'] >= target_success_rate]
    if eligible_rates:
        final_swr = max(r['swr'] for r in eligible_rates)
        initial_annual_withdrawal_amount = initial_investment * final_swr

    results_text = ""
    if final_swr > 0:
        results_text += f"The highest Safe Withdrawal Rate for {target_success_rate*100:.0f}% success over {num_years} years is approximately: {final_swr*100:.2f}%\n"
        results_text += f"This corresponds to an initial annual withdrawal of: ${initial_annual_withdrawal_amount:,.2f}\n"
    else:
        results_text += f"No tested SWR achieved the {target_success_rate*100:.0f}% success rate with the given assumptions.\n"
        lowest_tested_swr = min(r['swr'] for r in all_results)
        highest_success_at_lowest_swr = [r['success_rate'] for r in all_results if r['swr'] == lowest_tested_swr][0]
        results_text += f"The lowest tested SWR ({lowest_tested_swr*100:.1f}%) had a success rate of {highest_success_at_lowest_swr*100:.2f}%.\n"
        results_text += "Consider revising assumptions (e.g., higher returns, lower volatility/inflation, shorter horizon) or target success rate.\n"

    results_text += "\n--- All Tested Withdrawal Rates and Success Probabilities ---\n"
    for r in all_results:
        results_text += f"SWR: {r['swr']*100:.2f}% -> Success Rate: {r['success_rate']*100:.2f}%\n"

    # --- Plotting Results ---
    swrs_plot = [r['swr'] * 100 for r in all_results]
    success_rates_plot = [r['success_rate'] * 100 for r in all_results]

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(swrs_plot, success_rates_plot, marker='o', linestyle='-')
    ax1.axhline(y=target_success_rate * 100, color='r', linestyle='--', label=f'{target_success_rate*100:.0f}% Target Success')
    if final_swr > 0:
        ax1.axvline(x=final_swr * 100, color='g', linestyle=':', label=f'Calculated SWR: {final_swr*100:.2f}%')
    ax1.set_title(f'Monte Carlo SWR Success Rates ({num_simulations:,} simulations)')
    ax1.set_xlabel('Initial Withdrawal Rate (%)')
    ax1.set_ylabel('Probability of Portfolio Lasting 30 Years (%)')
    ax1.grid(True)
    ax1.legend()
    
    # Dynamically adjust y-axis limits for SWR Success Rates Plot
    if success_rates_plot:
        min_success = min(success_rates_plot)
        max_success = max(success_rates_plot)
        # Add a small buffer to the min/max for better visualization
        ax1.set_ylim(max(0, min_success - 5), min(100, max_success + 5))
    else:
        ax1.set_ylim(0, 105) # Fallback if no success rates are plotted

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    chosen_swr_for_path_plot = final_swr if final_swr > 0 else 0.035
    if chosen_swr_for_path_plot in portfolio_paths_for_plotting and portfolio_paths_for_plotting[chosen_swr_for_path_plot]:
        paths_to_plot_sample = portfolio_paths_for_plotting[chosen_swr_for_path_plot]
        for i, path_data in enumerate(paths_to_plot_sample):
            if len(path_data) == num_years + 1:
                 ax2.plot(range(num_years + 1), path_data, alpha=0.1, color='blue' if path_data[-1] > 0 else 'red')
        
        ax2.set_title(f'Sample Portfolio Paths for {chosen_swr_for_path_plot*100:.2f}% SWR (100 simulations shown)')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.set_yscale('log')
        ax2.grid(True, which="both", ls="-", alpha=0.5)
        ax2.axhline(y=initial_investment, color='k', linestyle='--', label=f'Initial: ${initial_investment:,.0f}')
        ax2.axhline(y=1, color='grey', linestyle=':', label='$1 (for log scale visibility near zero)')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, f"No portfolio paths were stored for SWR {chosen_swr_for_path_plot*100:.2f}% to plot individual simulations.",
                 horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Sample Portfolio Paths")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Portfolio Value ($)")

    # Save results to cache before returning
    _save_to_cache(cache_key, results_text, fig1, fig2)

    return f"Simulation Complete! Results text length: {len(results_text)}", results_text, fig1, fig2

# Gradio Interface
# Explanation text for the modal
explanation_text = """
---
## Understanding the Safe Withdrawal Rate (SWR) Calculator

### 1. What is the Safe Withdrawal Rate (SWR)?
The Safe Withdrawal Rate (SWR) is a concept primarily used in retirement planning. It refers to the percentage of your initial retirement portfolio that you can withdraw each year, adjusted for inflation, without running out of money over a specified period (e.g., 30 years). The goal is to find a withdrawal rate that has a very high probability of success, even in adverse market conditions.

This calculator uses a **Monte Carlo Simulation** approach to determine the SWR. Instead of relying on historical averages, which might not repeat, Monte Carlo simulations run thousands of possible future market scenarios based on statistical distributions (mean and standard deviation) of asset returns and inflation. For each scenario, it checks if the portfolio lasts the entire retirement period.

**Performance Note**: The simulation can be computationally intensive. To speed up calculations, consider reducing the `Number of Simulations` and `Number of SWR Intervals`. Higher values provide more accuracy but take longer to compute.

### 2. Key Assumptions Used in This Calculator and Their Meaning

The accuracy and relevance of the calculated SWR heavily depend on the assumptions you provide.

*   **Initial Investment**: Your starting portfolio value.
*   **Number of Years**: The duration of your retirement (e.g., 30 years).
*   **Number of Simulations**: How many different market scenarios the calculator runs. More simulations lead to more accurate results but take longer.
*   **Target Success Rate**: The probability you want your portfolio to last (e.g., 95% means 95 out of 100 simulations succeed).
*   **Stock Mean Return (%)**: The average annual nominal return you expect from your stock investments.
*   **Stock Standard Deviation (%)**: A measure of how much the stock returns are expected to vary from the mean (volatility). Higher standard deviation means more volatile returns.
*   **Bond Mean Return (%)**: The average annual nominal return you expect from your bond investments.
*   **Bond Standard Deviation (%)**: A measure of how much the bond returns are expected to vary from the mean (volatility).
*   **Correlation (Stocks vs. Bonds)**: How stock and bond returns move in relation to each other. A negative correlation means they tend to move in opposite directions, which can help diversify a portfolio.
*   **Stock Allocation (%)**: The percentage of your portfolio invested in stocks (the rest is in bonds).
*   **Mean Inflation (%)**: The average annual inflation rate you expect. Withdrawals are typically adjusted for inflation to maintain purchasing power.
*   **Inflation Standard Deviation (%)**: How much the inflation rate is expected to vary.
*   **SWR Test Range**: The range of withdrawal rates (e.g., 2.5% to 5.0%) that the simulation will test to find the optimal SWR.

### 3. Common Misunderstandings about SWR

*   **It's a Guarantee**: The SWR is a probability, not a guarantee. A 95% success rate means 5% of scenarios still fail.
*   **Fixed for Life**: The SWR is calculated based on initial assumptions. Life events, market shifts, or changes in spending can necessitate adjustments.
*   **One Size Fits All**: The SWR is highly personal and depends on your specific portfolio, risk tolerance, and spending habits.
*   **Only Initial Withdrawal Matters**: While the initial withdrawal rate is key, how you adjust withdrawals in response to market performance (e.g., reducing spending in down years) can significantly impact portfolio longevity. This calculator assumes inflation-adjusted withdrawals.

### 4. How the Calculator Works Internally

The core of this calculator is a Monte Carlo simulation that models thousands of possible future scenarios for your portfolio.

**Key Steps (Simplified):**

1.  **Parameter Initialization**: All input parameters are converted to decimal form (e.g., 9% becomes 0.09).
2.  **Covariance Matrix Calculation**: A covariance matrix is created using the mean returns, standard deviations, and correlation of stocks and bonds. This allows the simulation to generate realistic, correlated returns for both asset classes.
    ```python
    mean_asset_returns = np.array([stock_mean_return, bond_mean_return])
    cov_matrix = np.array([
        [stock_std_dev**2, correlation_stock_bond * stock_std_dev * bond_std_dev],
        [correlation_stock_bond * stock_std_dev * bond_std_dev, bond_std_dev**2]
    ])
    ```
3.  **Iterating SWRs**: The calculator tests a range of SWRs (e.g., from 2.5% to 5.0%).
4.  **Monte Carlo Loop (for each SWR)**:
    *   For each SWR, `num_simulations` (e.g., 10,000) independent scenarios are run.
    *   **Annual Simulation Loop**: For each year of the `num_years` retirement period:
        *   **Inflation Adjustment**: The annual withdrawal amount is adjusted for inflation based on a randomly generated inflation rate for that year.
            ```python
            yearly_inflation = np.random.normal(mean_inflation, std_dev_inflation)
            current_annual_withdrawal_nominal *= (1 + yearly_inflation)
            ```
        *   **Withdrawal**: The adjusted withdrawal amount is subtracted from the portfolio.
        *   **Ruin Check**: If the portfolio value drops to zero or below, the simulation for that scenario fails.
        *   **Generate Returns**: Random, correlated returns for stocks and bonds are generated for the year using the covariance matrix.
            ```python
            asset_returns_this_year = np.random.multivariate_normal(mean_asset_returns, cov_matrix)
            portfolio_return_this_year = (stock_allocation * asset_returns_this_year[0] +
                                          bond_allocation * asset_returns_this_year[1])
            ```
        *   **Apply Returns**: The portfolio value is updated with the generated returns.
    *   **Success/Failure Tracking**: After `num_years`, if the portfolio is still positive, the simulation is counted as a success.
5.  **Calculate Success Rate**: For each SWR, the `success_count` is divided by `num_simulations` to get the `success_probability`.
6.  **Find Optimal SWR**: The calculator then finds the highest SWR that meets or exceeds your `Target Success Rate`.
7.  **Plotting**: Finally, the results are visualized in two plots:
    *   **SWR Success Rates**: Shows the success probability curve across all tested SWRs.
    *   **Sample Portfolio Paths**: Displays a subset of individual simulation paths to illustrate portfolio behavior over time.
"""

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Safe Withdrawal Rate Calculator
        This application performs Monte Carlo simulations to determine a safe withdrawal rate from a retirement portfolio.
        Adjust the parameters below and click "Run Simulation" to see the results and portfolio projections.
        
        For a detailed explanation of SWR, assumptions, and internal workings, click the "Details" button below.
        """
    )
    
    with gr.Accordion("Details: Understanding the Safe Withdrawal Rate (SWR) Calculator", open=False):
        gr.Markdown(explanation_text)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Investment Details")
            initial_investment = gr.Number(label="Initial Investment ($)", value=1_000_000.0, interactive=True)
            num_years = gr.Slider(minimum=10, maximum=60, value=30, step=1, label="Number of Years", interactive=True)
            target_success_rate = gr.Slider(minimum=70, maximum=100, value=95, step=1, label="Target Success Rate (%)", interactive=True)
            num_simulations = gr.Slider(minimum=100, maximum=20000, value=5000, step=100, label="Number of Simulations", interactive=True)

            gr.Markdown("### Market Assumptions (Annualized Nominal Returns)")
            stock_mean_return = gr.Slider(minimum=0, maximum=20, value=9.0, step=0.1, label="Stock Mean Return (%)", interactive=True)
            stock_std_dev = gr.Slider(minimum=5, maximum=30, value=15.0, step=0.1, label="Stock Standard Deviation (%)", interactive=True)
            bond_mean_return = gr.Slider(minimum=0, maximum=10, value=4.0, step=0.1, label="Bond Mean Return (%)", interactive=True)
            bond_std_dev = gr.Slider(minimum=1, maximum=15, value=5.0, step=0.1, label="Bond Standard Deviation (%)", interactive=True)
            correlation_stock_bond = gr.Slider(minimum=-1.0, maximum=1.0, value=-0.2, step=0.01, label="Correlation (Stocks vs. Bonds)", interactive=True)
            stock_allocation = gr.Slider(minimum=0, maximum=100, value=60, step=1, label="Stock Allocation (%)", interactive=True)

            gr.Markdown("### Inflation Assumptions (Annualized)")
            mean_inflation = gr.Slider(minimum=0, maximum=10, value=2.5, step=0.1, label="Mean Inflation (%)", interactive=True)
            std_dev_inflation = gr.Slider(minimum=0, maximum=5, value=1.5, step=0.1, label="Inflation Standard Deviation (%)", interactive=True)
            
            gr.Markdown("### SWR Test Range")
            min_swr_test = gr.Slider(minimum=0.5, maximum=10.0, value=2.5, step=0.1, label="Min SWR to Test (%)", interactive=True)
            max_swr_test = gr.Slider(minimum=0.5, maximum=10.0, value=5.0, step=0.1, label="Max SWR to Test (%)", interactive=True)
            num_swr_intervals = gr.Slider(minimum=5, maximum=100, value=15, step=1, label="Number of SWR Intervals", interactive=True)

            run_button = gr.Button("Run Simulation", variant="primary")

        with gr.Column():
            gr.Markdown("### Simulation Results")
            status_output = gr.Textbox(label="Status", interactive=False, lines=1)
            
            gr.Markdown("#### Calculated Safe Withdrawal Rate Summary")
            results_output = gr.Textbox(label="Summary Text", interactive=False) # Removed lines=5 to allow auto-scrolling
            
            gr.Markdown("#### SWR Success Rates Plot")
            swr_plot_output = gr.Plot(label="SWR Success Rates")
            
            gr.Markdown("#### Sample Portfolio Paths Plot")
            paths_plot_output = gr.Plot(label="Sample Portfolio Paths")
            gr.Button("Buy Me a Coffee ☕", link="https://buymeacoffee.com/liaoch", variant="primary")

    run_button.click(
        fn=run_simulation,
        inputs=[
            initial_investment,
            num_years,
            num_simulations,
            target_success_rate,
            stock_mean_return,
            stock_std_dev,
            bond_mean_return,
            bond_std_dev,
            stock_allocation,
            correlation_stock_bond,
            mean_inflation,
            std_dev_inflation,
            min_swr_test,
            max_swr_test,
            num_swr_intervals
        ],
        outputs=[status_output, results_output, swr_plot_output, paths_plot_output]
    )

    # Define default parameters for initial load
    DEFAULT_PARAMS = {
        "initial_investment": 1_000_000.0,
        "num_years": 30,
        "num_simulations": 5000,
        "target_success_rate": 95,
        "stock_mean_return": 9.0,
        "stock_std_dev": 15.0,
        "bond_mean_return": 4.0,
        "bond_std_dev": 5.0,
        "stock_allocation": 60,
        "correlation_stock_bond": -0.2,
        "mean_inflation": 2.5,
        "std_dev_inflation": 1.5,
        "min_swr_test": 2.5,
        "max_swr_test": 5.0,
        "num_swr_intervals": 15
    }

    def load_default_results():
        """Loads cached results for default parameters if available."""
        # Ensure the order of parameters matches _generate_cache_key
        default_key = _generate_cache_key(
            DEFAULT_PARAMS["initial_investment"],
            DEFAULT_PARAMS["num_years"],
            DEFAULT_PARAMS["num_simulations"],
            DEFAULT_PARAMS["target_success_rate"],
            DEFAULT_PARAMS["stock_mean_return"],
            DEFAULT_PARAMS["stock_std_dev"],
            DEFAULT_PARAMS["bond_mean_return"],
            DEFAULT_PARAMS["bond_std_dev"],
            DEFAULT_PARAMS["stock_allocation"],
            DEFAULT_PARAMS["correlation_stock_bond"],
            DEFAULT_PARAMS["mean_inflation"],
            DEFAULT_PARAMS["std_dev_inflation"],
            DEFAULT_PARAMS["min_swr_test"],
            DEFAULT_PARAMS["max_swr_test"],
            DEFAULT_PARAMS["num_swr_intervals"]
        )
        
        cached = _load_from_cache(default_key)
        if cached:
            return cached
        else:
            # Return empty/placeholder values if no cache hit
            return "No default cache found. Run simulation.", "", None, None

    # Load default results on app startup
    demo.load(
        fn=load_default_results,
        outputs=[status_output, results_output, swr_plot_output, paths_plot_output]
    )

    demo.launch()
