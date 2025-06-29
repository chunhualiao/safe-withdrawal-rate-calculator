import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

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
    swr_test_step: float,
    progress=gr.Progress()
):
    progress(0, desc="Starting simulation...")
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
    withdrawal_rates_to_test = np.arange(min_swr_test / 100.0, (max_swr_test + swr_test_step) / 100.0, swr_test_step / 100.0)
    all_results = []
    portfolio_paths_for_plotting = {}

    for idx, swr in enumerate(withdrawal_rates_to_test):
        progress((idx + 1) / len(withdrawal_rates_to_test), desc=f"Simulating SWR: {swr*100:.1f}%")
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
    ax1.set_ylim(0, 105)

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

    return "Simulation Complete!", results_text, fig1, fig2

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Safe Withdrawal Rate Calculator
        This application performs Monte Carlo simulations to determine a safe withdrawal rate from a retirement portfolio.
        Adjust the parameters below and click "Run Simulation" to see the results and portfolio projections.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Investment Details")
            initial_investment = gr.Number(label="Initial Investment ($)", value=1_000_000.0, interactive=True)
            num_years = gr.Slider(minimum=10, maximum=60, value=30, step=1, label="Number of Years", interactive=True)
            target_success_rate = gr.Slider(minimum=70, maximum=100, value=95, step=1, label="Target Success Rate (%)", interactive=True)
            num_simulations = gr.Slider(minimum=1000, maximum=50000, value=10000, step=1000, label="Number of Simulations", interactive=True)

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
            swr_test_step = gr.Slider(minimum=0.01, maximum=0.5, value=0.1, step=0.01, label="SWR Test Step (%)", interactive=True)

            run_button = gr.Button("Run Simulation")

        with gr.Column():
            gr.Markdown("### Simulation Results")
            status_output = gr.Textbox(label="Status", interactive=False, lines=1)
            
            gr.Markdown("#### Calculated Safe Withdrawal Rate Summary")
            results_output = gr.Textbox(label="Summary Text", lines=5)
            
            gr.Markdown("#### SWR Success Rates Plot")
            swr_plot_output = gr.Plot(label="SWR Success Rates")
            
            gr.Markdown("#### Sample Portfolio Paths Plot")
            paths_plot_output = gr.Plot(label="Sample Portfolio Paths")

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
            swr_test_step
        ],
        outputs=[status_output, results_output, swr_plot_output, paths_plot_output]
    )

demo.launch()
