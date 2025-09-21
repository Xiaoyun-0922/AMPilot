"""
Statistical Analysis Templates for Data Analysis Agent

Templates and result interpretations for common statistical analysis methods
"""

# Correlation analysis template
CORRELATION_ANALYSIS_TEMPLATE = """
## Correlation Analysis Result Interpretation

### 1. Pearson Correlation
- **Applicable conditions**: Continuous variables, linear relationship, normally distributed data
- **Coefficient range**: -1 to 1
- **Interpretation standards**:
  - |r| ≥ 0.8: Strong correlation
  - 0.5 ≤ |r| < 0.8: Moderate correlation
  - 0.3 ≤ |r| < 0.5: Weak correlation
  - |r| < 0.3: Almost no correlation

### 2. Spearman Correlation
- **Applicable conditions**: Ordinal variables, monotonic relationship, no normality requirement
- **Advantages**: Insensitive to outliers, suitable for non-linear monotonic relationships

### 3. Kendall's Tau
- **Applicable conditions**: Small samples, ordinal data
- **Characteristics**: More robust, but computationally complex

### Key interpretation points:
1. **Significance level**: p < 0.05 indicates significant correlation
2. **Effect size**: Consider both significance and correlation coefficient magnitude
3. **Causality**: Correlation does not imply causation, requires domain knowledge
"""

# Hypothesis testing template
HYPOTHESIS_TEST_TEMPLATE = """
## Hypothesis Testing Result Interpretation

### 1. t-test
#### One-sample t-test
- **Purpose**: Test whether sample mean equals a specific value
- **Hypothesis**: H0: μ = μ0, H1: μ ≠ μ0
- **Applicable conditions**: Data approximately normally distributed

#### Independent samples t-test
- **Purpose**: Compare mean differences between two independent groups
- **Hypothesis**: H0: μ1 = μ2, H1: μ1 ≠ μ2
- **Applicable conditions**: Two groups independent, approximately normal, equal variances

#### Paired samples t-test
- **Purpose**: Compare differences in the same subjects under two conditions
- **Hypothesis**: H0: μd = 0, H1: μd ≠ 0
- **Applicable conditions**: Paired data, differences approximately normal

### 2. Analysis of Variance (ANOVA)
#### One-way ANOVA
- **Purpose**: Compare mean differences among three or more groups
- **Hypothesis**: H0: μ1 = μ2 = ... = μk, H1: At least one pair of means differs
- **Applicable conditions**: Groups independent, normal distribution, equal variances

#### Two-way ANOVA
- **Purpose**: Analyze effects of two factors on dependent variable and their interaction
- **Advantages**: Can test main effects and interaction effects

### 3. Chi-square test
- **Purpose**: Test association between categorical variables
- **Hypothesis**: H0: Variables are independent, H1: Variables are associated
- **Applicable conditions**: Expected frequencies ≥ 5

### Key interpretation points:
1. **p-value**: p < 0.05 reject null hypothesis, p ≥ 0.05 fail to reject null hypothesis
2. **Effect size**: Cohen's d, eta squared, etc.
3. **Confidence interval**: Provides uncertainty range for parameter estimates
4. **Statistical power**: Ability to detect true effects
"""

# Regression analysis template
REGRESSION_ANALYSIS_TEMPLATE = """
## Regression Analysis Result Interpretation

### 1. Linear Regression
#### Model evaluation metrics:
- **R²**: Coefficient of determination, proportion of variance explained (0-1)
  - R² > 0.7: Good model fit
  - 0.3 < R² < 0.7: Moderate model fit
  - R² < 0.3: Poor model fit
- **Adjusted R²**: R² adjusted for number of variables
- **F-statistic**: Overall model significance test
- **AIC/BIC**: Model selection criteria, lower is better

#### Regression coefficient interpretation:
- **Intercept**: Expected value of dependent variable when all predictors are 0
- **Slope**: Average change in dependent variable per unit increase in predictor
- **Standard error**: Uncertainty in coefficient estimate
- **t-value and p-value**: Coefficient significance test

#### Model assumption testing:
1. **Linearity**: Residuals should be randomly distributed
2. **Independence**: No autocorrelation among residuals
3. **Homoscedasticity**: Constant residual variance
4. **Normality**: Residuals approximately normally distributed
5. **No multicollinearity**: VIF < 10

### 2. Logistic Regression
- **Purpose**: Predict probability of binary outcomes
- **Odds Ratio (OR)**: exp(β), effect of predictor on odds of outcome
- **Model evaluation**: AUC, accuracy, sensitivity, specificity

### Key interpretation points:
1. **Coefficient significance**: p < 0.05 indicates significant effect on outcome
2. **Model fit**: Evaluated through R², AIC and other metrics
3. **Predictive ability**: Assessed through cross-validation methods
4. **Practical significance**: Interpret coefficients in professional context
"""

# Descriptive statistics template
DESCRIPTIVE_STATS_TEMPLATE = """
## Descriptive Statistics Result Interpretation

### 1. Central tendency measures
- **Mean**: Average level of data, sensitive to outliers
- **Median**: Middle value of data, insensitive to outliers
- **Mode**: Most frequently occurring value

### 2. Dispersion measures
- **Standard deviation (SD)**: Average dispersion around the mean
- **Variance**: Square of standard deviation
- **Interquartile range (IQR)**: Q3 - Q1, range of middle 50% of data
- **Coefficient of variation (CV)**: SD/Mean, relative dispersion

### 3. Distribution shape measures
- **Skewness**:
  - > 0: Right-skewed (positive skew)
  - < 0: Left-skewed (negative skew)
  - = 0: Symmetric distribution
- **Kurtosis**:
  - > 0: Leptokurtic (peaked)
  - < 0: Platykurtic (flat)
  - = 0: Normal distribution

### 4. Outlier detection
- **Box plot method**: Points beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR
- **Z-score method**: Points with |Z| > 3
- **IQR method**: Interquartile range method

### Key interpretation points:
1. **Data quality**: Check for missing values and outliers
2. **Distribution characteristics**: Understand basic distribution patterns
3. **Variable relationships**: Preliminary exploration of relationships
4. **Further analysis**: Provide foundation for subsequent analysis
"""

# Survival analysis template
SURVIVAL_ANALYSIS_TEMPLATE = """
## Survival Analysis Result Interpretation

### 1. Kaplan-Meier survival curves
- **Survival function**: S(t) = P(T > t), probability of survival beyond time t
- **Median survival time**: Time point where survival probability is 50%
- **Survival rate**: Survival probability at specific time points

### 2. Log-rank test
- **Purpose**: Compare survival curve differences between groups
- **Hypothesis**: H0: Survival curves are identical across groups
- **Applicable**: When proportional hazards assumption holds

### 3. Cox proportional hazards model
- **Hazard ratio (HR)**: exp(β), relative risk
  - HR > 1: Increased risk
  - HR < 1: Decreased risk
  - HR = 1: No effect
- **95% confidence interval**: Uncertainty range for HR

### Key interpretation points:
1. **Censored data**: Properly handle data where events are not observed
2. **Proportional hazards assumption**: Test whether model assumptions hold
3. **Clinical significance**: Interpret results in medical context
"""

# Get template function
def get_analysis_template(analysis_type: str) -> str:
    """Get analysis template"""
    templates = {
        'correlation': CORRELATION_ANALYSIS_TEMPLATE,
        'hypothesis_test': HYPOTHESIS_TEST_TEMPLATE,
        'regression': REGRESSION_ANALYSIS_TEMPLATE,
        'descriptive': DESCRIPTIVE_STATS_TEMPLATE,
        'survival': SURVIVAL_ANALYSIS_TEMPLATE
    }

    return templates.get(analysis_type.lower(), "Template not found for the specified analysis type")

# All available analysis types
AVAILABLE_ANALYSIS_TYPES = [
    'correlation',      # Correlation analysis
    'hypothesis_test',  # Hypothesis testing
    'regression',       # Regression analysis
    'descriptive',      # Descriptive statistics
    'survival'          # Survival analysis
]
