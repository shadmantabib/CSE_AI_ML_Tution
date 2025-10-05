# Batch Gradient Descent (Detailed Numeric Example)

This document explains **batch gradient descent** step-by-step for a
linear regression model with\
**m = 3 training examples** and **n = 2 features**.

------------------------------------------------------------------------

## Step 0: Setup

We use a linear model:

\[ h\_`\theta`{=tex}(x) = `\theta`{=tex}\_0 + `\theta`{=tex}\_1 x_1 +
`\theta`{=tex}\_2 x_2 \]

### Training Data

  Example   x₁   x₂   y
  --------- ---- ---- ---
  1         1    0    3
  2         0    1    4
  3         1    1    5

These follow approximately: ( y = 2 + 1x_1 + 2x_2 ).

**Initial Parameters:**\
( `\theta`{=tex}\_0 = 0, `\theta`{=tex}\_1 = 0, `\theta`{=tex}\_2 = 0 )

**Learning Rate:**\
( `\alpha `{=tex}= 0.1 )

------------------------------------------------------------------------

## Step 1: Initial Predictions

  i   x₁   x₂   y   hθ(x)   Error (hθ - y)
  --- ---- ---- --- ------- ----------------
  1   1    0    3   0       -3
  2   0    1    4   0       -4
  3   1    1    5   0       -5

------------------------------------------------------------------------

## Step 2: Gradient Calculation

Formulas:

\[ `\frac{\partial J}{\partial \theta_j}`{=tex} =
`\frac{1}{m}`{=tex}`\sum`{=tex}\_{i=1}^{m}(h\_`\theta`{=tex}(x^{(i)}) -
y\^{(i)}) x_j\^{(i)} \]

### Gradients

( `\frac{\partial J}{\partial \theta_0}`{=tex} = -4.0 )\
( `\frac{\partial J}{\partial \theta_1}`{=tex} = -2.67 )\
( `\frac{\partial J}{\partial \theta_2}`{=tex} = -3.0 )

------------------------------------------------------------------------

## Step 3: Parameter Update

\[ `\theta`{=tex}\_j := `\theta`{=tex}\_j -
`\alpha `{=tex}`\times `{=tex}`\text{gradient}`{=tex}\_j \]

  Parameter   Before   Update   After
  ----------- -------- -------- -------
  θ₀          0.0      +0.4     0.4
  θ₁          0.0      +0.267   0.267
  θ₂          0.0      +0.3     0.3

✅ **After 1st iteration:** ( `\theta `{=tex}= (0.4, 0.267, 0.3) )

------------------------------------------------------------------------

## Step 4: Compute Cost (J(`\theta`{=tex}))

\[ J(`\theta`{=tex}) =
`\frac{1}{2m}`{=tex}`\sum`{=tex}\_{i=1}^{m}(h\_`\theta`{=tex}(x^{(i)}) -
y^{(i)})^2 \]

  i   x₁   x₂   y   hθ(x)   Error    Error²
  --- ---- ---- --- ------- -------- --------
  1   1    0    3   0.667   -2.333   5.44
  2   0    1    4   0.700   -3.300   10.89
  3   1    1    5   0.967   -4.033   16.27

( `\text{Sum(Error²)}`{=tex} = 32.6 )

\[ J(`\theta`{=tex}) = `\frac{1}{2\times3}`{=tex}(32.6) = 5.43 \]

✅ **Cost after 1st iteration:** 5.43

------------------------------------------------------------------------

## Step 5: 2nd Iteration

### New Predictions

  i   x₁   x₂   y   hθ(x)   Error
  --- ---- ---- --- ------- --------
  1   1    0    3   0.667   -2.333
  2   0    1    4   0.700   -3.300
  3   1    1    5   0.967   -4.033

### Gradients

  Parameter   Gradient
  ----------- ----------
  θ₀          -3.22
  θ₁          -2.12
  θ₂          -2.44

### Updates

  Parameter   Before   Update   After
  ----------- -------- -------- -------
  θ₀          0.4      +0.322   0.722
  θ₁          0.267    +0.212   0.479
  θ₂          0.3      +0.244   0.544

✅ **After 2nd iteration:** ( `\theta `{=tex}= (0.722, 0.479, 0.544) )

------------------------------------------------------------------------

## Step 6: Cost After 2nd Iteration

  i   x₁   x₂   y   hθ(x)   Error    Error²
  --- ---- ---- --- ------- -------- --------
  1   1    0    3   1.201   -1.799   3.24
  2   0    1    4   1.267   -2.733   7.48
  3   1    1    5   1.746   -3.254   10.60

Sum(Error²) = 21.32\
\[ J = `\frac{1}{2\times3}`{=tex}(21.32) = 3.55 \]

✅ **Cost after 2nd iteration:** 3.55

------------------------------------------------------------------------

## Step 7: 3rd Iteration (Summary)

After one more iteration:

( `\theta `{=tex}= (0.982, 0.647, 0.744) )\
( J(`\theta`{=tex}) = 2.06 )

------------------------------------------------------------------------

## Convergence Table

  Iter   θ₀      θ₁      θ₂      J(θ)
  ------ ------- ------- ------- ------
  0      0.000   0.000   0.000   8.33
  1      0.400   0.267   0.300   5.43
  2      0.722   0.479   0.544   3.55
  3      0.982   0.647   0.744   2.06
  4      1.19    0.781   0.907   1.22
  5      1.36    0.886   1.04    0.83

------------------------------------------------------------------------

## Summary

-   **Gradient Descent** uses all 3 samples each iteration.\
-   The **cost J(θ)** decreases steadily.\
-   Parameters converge close to the true values **(2, 1, 2)**.\
-   Larger α → faster but risk divergence; smaller α → slower but
    stable.

------------------------------------------------------------------------

**End of Example.**
