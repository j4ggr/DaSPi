# Hypothesis Testing

Statistical hypothesis tests provide a rigorous framework for deciding whether an observed pattern in data is real or could have occurred by chance. DaSPi bundles a curated set of tests — covering normality, variance equality, location differences, proportions, and distribution shape — with a **consistent return signature**: every test function returns a tuple whose **first element is always the p-value**, making it straightforward to integrate them into automated pipelines.

## Key Concepts

### p-value and Significance Level

The **p-value** is the probability of observing data at least as extreme as the sample, assuming the null hypothesis ($H_0$) is true.

- If $p \leq \alpha$ → **reject** $H_0$ (statistically significant)
- If $p > \alpha$ → **fail to reject** $H_0$

The significance level $\alpha$ is typically set at $0.05$ (5 %). Use `confidence_to_alpha` to convert a confidence level to $\alpha$:

```python
from daspi.statistics import confidence_to_alpha

alpha = confidence_to_alpha(0.95)   # → 0.05
```

---

## Normality Tests

Many parametric tests assume normally distributed data. Always check normality first.

### Anderson-Darling Test

The recommended test for both small and large sample sizes — it weights deviations in the tails more heavily than a simple KS test.

```python
import daspi as dsp

sample = dsp.load_dataset('anova')['Pain threshold']

p, A2 = dsp.anderson_darling_test(sample)
print(f"p = {p:.4f}, A² = {A2:.4f}")
# p > 0.05 → normal distribution not rejected
```

The statistic $A^2$ is compared against critical values derived from the theoretical normal distribution. The returned p-value is an interpolated continuous approximation.

### All-Normal Convenience Check

Test multiple groups at once:

```python
df = dsp.load_dataset('anova')

groups = [group['Pain threshold'].values
          for _, group in df.groupby('Hair color')]

if dsp.all_normal(*groups):
    print("All groups appear normally distributed → parametric tests are valid")
else:
    print("At least one group is non-normal → consider non-parametric alternatives")
```

### Kolmogorov-Smirnov Test

One-sample KS test against any continuous SciPy distribution:

```python
from scipy.stats import norm
import daspi as dsp

sample = dsp.load_dataset('anova')['Pain threshold']

# Test against a standard normal (shift and scale the data first)
p, D = dsp.kolmogorov_smirnov_test(sample, dist=norm)
print(f"KS: p = {p:.4f}, D = {D:.4f}")
```

---

## Variance (Homogeneity) Tests

### F-Test

Compares variances of two independent, *normally distributed* samples:

$$F = \frac{s_1^2}{s_2^2}$$

```python
p, F = dsp.f_test(group_a, group_b)
print(f"F-test: p = {p:.4f}, F = {F:.4f}")
```

!!! warning "Assumption"
    The F-test assumes both samples are normally distributed. Use `levene_test` for non-normal data.

### Levene Test

Robust alternative that does not require normality:

```python
p, W = dsp.levene_test(group_a, group_b)
print(f"Levene: p = {p:.4f}, W = {W:.4f}")

# For heavy-tailed (leptokurtic) distributions
p, W = dsp.levene_test(group_a, group_b, heavy_tailed=True)
```

### Auto-Select with `variance_test`

Let DaSPi choose the appropriate test based on normality:

```python
p, stat = dsp.variance_test(group_a, group_b)
# Runs the F-test if both groups are normal, Levene otherwise
```

### Variance Stability (Single Sample)

Check whether the variance of a *single* time-ordered sample remains constant — useful for verifying process stability before capability analysis:

```python
p, W = dsp.variance_stability_test(sample, n_sections=3)
# Splits sample into 3 sections and applies Levene across them
```

---

## Location (Mean / Median) Tests

### One-Sample t-Test

Test whether the sample mean equals a hypothesised population mean $\mu_0$:

```python
p, t, df = dsp.t_test(sample, mu=100)
print(f"t-test: p = {p:.4f}, t = {t:.4f}, df = {df}")
```

### Two-Sample Location Test (`position_test`)

Automatically dispatches to the independent-samples **t-test** (parametric) or **Mann-Whitney U** (non-parametric) based on normality and variance equality:

```python
import daspi as dsp

df = dsp.load_dataset('anova')
group_a = df.loc[df['Hair color'] == 'Light Blond', 'Pain threshold']
group_b = df.loc[df['Hair color'] == 'Dark Brunette', 'Pain threshold']

p, stat = dsp.position_test(group_a, group_b)
print(f"Location test: p = {p:.4f}")
```

Selection logic:

| Both normal | Equal variances | Test used |
|:-----------:|:---------------:|-----------|
| Yes | Yes | Student's t-test (pooled) |
| Yes | No  | Welch's t-test |
| No  | —   | Mann-Whitney U |

### Mean Stability (Single Sample)

One-way ANOVA across time-ordered sections — lightweight control-chart alternative:

```python
p, F = dsp.mean_stability_test(sample, n_sections=3)
# p < 0.05 suggests the mean is shifting over time
```

---

## Proportion Tests

Compare a proportion between two independent groups:

```python
# counts of events, total observations per group
p, stat = dsp.proportions_test(count1=12, nobs1=50, count2=7, nobs2=50)
print(f"Proportions test: p = {p:.4f}")
```

For small samples (expected cell count < 5) Fisher's exact test is used automatically.

---

## Shape Tests

### Skewness

```python
p, Z = dsp.skew_test(sample)
# H0: population skewness is zero
```

### Kurtosis (Excess)

```python
p, Z = dsp.kurtosis_test(sample)
# H0: population excess kurtosis is zero
```

---

## Worked Example: Complete Two-Sample Comparison

```python
import daspi as dsp

df = dsp.load_dataset('tips')
lunch  = df.loc[df['time'] == 'Lunch',  'total_bill']
dinner = df.loc[df['time'] == 'Dinner', 'total_bill']

# 1. Normality
p_norm_l, _ = dsp.anderson_darling_test(lunch)
p_norm_d, _ = dsp.anderson_darling_test(dinner)
both_normal  = p_norm_l > 0.05 and p_norm_d > 0.05

# 2. Variance equality (only relevant if both normal)
p_var, _ = dsp.variance_test(lunch, dinner)
equal_var = p_var > 0.05

# 3. Location test (auto-selects)
p_loc, stat = dsp.position_test(lunch, dinner)

print(f"Normal: {both_normal}, Equal variance: {equal_var}")
print(f"Location p = {p_loc:.4f} → "
      f"{'different' if p_loc < 0.05 else 'no significant difference'}")
```

---

## Integrating Tests with ANOVA

Hypothesis tests are also run internally by `LinearModel` (via the
`GageStudyModel` and `LocationDispersionEstimator`) to validate model
assumptions. See the [ANOVA Guide](anova.md) and the
[API reference](../statistics/hypothesis/index.md) for details.
