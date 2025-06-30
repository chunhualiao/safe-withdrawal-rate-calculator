---
title: Safe Withdrawal Rate Calculator
emoji: 🏢
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
license: mit
short_description: Safe Withdrawal Rate Calculator
---

![screenshot](https://private-user-images.githubusercontent.com/1627206/460409282-a733a6d3-4b48-4e02-b3ab-2e11efc33c15.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTEyNTg0NDQsIm5iZiI6MTc1MTI1ODE0NCwicGF0aCI6Ii8xNjI3MjA2LzQ2MDQwOTI4Mi1hNzMzYTZkMy00YjQ4LTRlMDItYjNhYi0yZTExZWZjMzNjMTUucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI1MDYzMCUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTA2MzBUMDQzNTQ0WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9OGQyYWIyNzJjYTY4ZWU5MDljMjZmNDBlNDVjMjY1ZTE2M2MyMGMxM2U0N2M5OWFhNDMwMWMwMTJiMWM2YzRjNCZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.eG9Qe-6Wlvg-A0OlJV4ycz6zPFJjnJQVOKxRotoLgPw)

This application is deployed to [Hugging Face Spaces](https://huggingface.co/spaces/liaoch/Safe-Withdrawal-Rate-Calculator).

This Hugging Face Space hosts an interactive application that performs Monte Carlo simulations to determine a safe withdrawal rate from a retirement portfolio. Users can adjust various financial parameters and instantly see the impact on portfolio success rates and projected portfolio paths.

## How to Use

1.  **Adjust Parameters**: On the left side of the interface, you will find several sections with sliders and number inputs:
    *   **Investment Details**: Set your initial investment, the number of years for the simulation, your target success rate, and the number of Monte Carlo simulations to run.
    *   **Market Assumptions**: Define the expected mean returns and standard deviations for stocks and bonds, their correlation, and your portfolio's stock allocation.
    *   **Inflation Assumptions**: Input the mean inflation rate and its standard deviation.
    *   **SWR Test Range**: Specify the range and step for the Safe Withdrawal Rates you want to test.
2.  **Run Simulation**: Click the "Run Simulation" button.
3.  **View Results**: The right side of the interface will display:
    *   **Calculated Safe Withdrawal Rate**: Text output showing the highest SWR that meets your target success rate and the corresponding initial annual withdrawal amount.
    *   **SWR Success Rates Plot**: A graph showing the probability of portfolio success for various withdrawal rates.
    *   **Sample Portfolio Paths Plot**: A visualization of how a sample of portfolios might perform over the simulation period.

## Technical Details

This application is built using:

*   **Python**: For the core simulation logic.
*   **NumPy**: For numerical operations and statistical calculations.
*   **Matplotlib**: For generating the simulation plots.
*   **Gradio**: For creating the interactive web interface, allowing easy deployment to Hugging Face Spaces.

## Deployment

This application is designed to be deployed on Hugging Face Spaces. The `app.py` file contains the Gradio application, and `requirements.txt` lists all necessary Python dependencies.
