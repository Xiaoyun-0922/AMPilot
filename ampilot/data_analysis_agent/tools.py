import pandas as pd
# ---vvv--- START OF FIX ---vvv---
import matplotlib
# We must set the backend before importing pyplot
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# ---^^^--- END OF FIX ---^^^---
from langchain_core.tools import tool
from statsmodels.stats.weightstats import ztest
import statsmodels.api as sm

# ... a többi tool (hypothesis_test_z_test, regression_analysis) változatlan marad ...
@tool
def hypothesis_test_z_test(csv_path: str, column1_name: str, column2_name: str) -> str:
    """
    Performs a two-sample Z-test on two columns from a given CSV file.
    This tool is used to determine if two independent samples were drawn from populations with the same mean.
    Requires the file path of the CSV and the names of the two columns to compare.
    """
    try:
        df = pd.read_csv(csv_path)
        group1 = df[column1_name].dropna()
        group2 = df[column2_name].dropna()
        
        z_stat, p_value = ztest(group1, group2, value=0)
        
        alpha = 0.05
        result = f"Z-Test Results:\n"
        result += f"  - Z-statistic: {z_stat:.4f}\n"
        result += f"  - P-value: {p_value:.4f}\n\n"
        
        if p_value < alpha:
            result += f"Conclusion: The p-value ({p_value:.4f}) is less than the significance level {alpha}, so we reject the null hypothesis.\n"
            result += f"This suggests a statistically significant difference between the means of '{column1_name}' and '{column2_name}'."
        else:
            result += f"Conclusion: The p-value ({p_value:.4f}) is greater than or equal to the significance level {alpha}, so we fail to reject the null hypothesis.\n"
            result += f"There is not enough evidence to suggest a significant difference between the means of '{column1_name}' and '{column2_name}'."
            
        return result
    except Exception as e:
        return f"An error occurred during the Z-test: {e}"

@tool
def regression_analysis(csv_path: str, independent_var_column: str, dependent_var_column: str) -> str:
    """
    Performs a simple linear regression analysis on two columns from a given CSV file.
    This tool determines how an independent variable influences a dependent variable.
    """
    try:
        df = pd.read_csv(csv_path)
        
        y = df[dependent_var_column]
        X = df[independent_var_column]
        X = sm.add_constant(X) # Add a constant (intercept) to the model
        
        model = sm.OLS(y, X).fit()
        
        return f"Linear regression analysis complete. Here is the model summary:\n\n{str(model.summary())}"
    except Exception as e:
        return f"An error occurred during regression analysis: {e}"

@tool
def visualize_data_curve(csv_path: str, columns_to_plot: list[str], x_axis_column: str = None, output_image_path: str = 'data_curve.png') -> str:
    """
    Plots specified columns from a CSV file as curves, like a time series.
    You can specify a column for the x-axis, or it will use the index by default.
    The chart is saved to a local image file.
    """
    try:
        # The fix is at the top of the file, this function's code does not need to change.
        df = pd.read_csv(csv_path)
        
        plt.figure(figsize=(12, 6))
        
        x_data = df[x_axis_column] if x_axis_column and x_axis_column in df.columns else df.index
        
        for col in columns_to_plot:
            if col in df.columns:
                plt.plot(x_data, df[col], label=col)
            else:
                return f"Warning: Column '{col}' not found in the CSV file. Skipping."
        
        plt.title('Data Visualization')
        plt.xlabel(x_axis_column if x_axis_column else 'Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(output_image_path)
        plt.close() # Good practice to close the plot to free memory
        
        return f"Chart successfully generated and saved to: '{output_image_path}'"
    except Exception as e:
        return f"An error occurred during data visualization: {e}"