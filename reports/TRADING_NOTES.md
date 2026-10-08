# Trading notes - Black-Litterman factor screen, October 07, 2026

Each note below explains *why* a stock is in the portfolio using statistics, and what would make us sell it.
Model capital: $100,000. All statistics use the trailing 252 trading days unless stated.

## Why trust the screen at all (strategy-level evidence)

Walk-forward test Feb 2013 to Oct 2026, monthly rebalancing, 10 bp trading costs, no look-ahead:

- Return 20.9%/yr vs SPY 14.8%/yr; Sharpe 1.04 vs 0.80.
- Alpha vs SPY 6.8%/yr, t = 3.73; still significant after a Holm correction for testing 14 methods (adjusted p = 0.003). Bayes factor 30 under a skeptical prior (alpha ~ N(0, 2%^2)).
- Alpha vs an equal-weight portfolio of the same stocks (removes survivorship and size effects): 4.0%/yr, t = 2.40, posterior P(alpha > 0) = 97%. This is suggestive, not proof: after the Holm correction p = 0.23.
- Not sensitive to the Black-Litterman settings (tau, IC); removing the factor views turns alpha negative, so the views are what add value.

## Market regime check

A two-state hidden Markov model fitted to three years of S&P 500 daily returns puts the probability that we are in the high-volatility regime at **1%** (calm). Positions are taken at full size.

## Summary

| Ticker | Sector | Verdict | Weight | $ | Shares | Price | Stop | BL prior → posterior | P(drift>0) | Beta | α t-stat |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| MPC | Energy | BUY | 10.0% | $10,000 | 22 | $442.26 | $351.74 | 0.4% → 1.7% | 97% | -0.13 | 2.33 |
| PSX | Energy | BUY | 10.0% | $10,000 | 36 | $271.62 | $222.49 | 0.5% → 1.7% | 98% | -0.23 | 2.44 |
| MSFT | Information Technology | BUY - half size | 4.2% | $4,202 | 7 | $529.76 | $429.85 | 1.5% → 1.2% | 85% | 0.96 | -0.32 |
| KO | Consumer Staples | BUY | 8.0% | $8,037 | 93 | $85.82 | $76.54 | 0.9% → 1.0% | 96% | -0.24 | 1.63 |
| AMD | Information Technology | BUY - half size | 3.5% | $3,535 | 5 | $645.86 | $388.61 | 4.1% → 5.8% | 92% | 3.17 | 1.79 |
| JNJ | Health Care | BUY | 6.0% | $5,996 | 23 | $258.45 | $229.89 | 1.1% → 1.3% | 97% | -0.18 | 1.66 |
| MU | Information Technology | BUY - half size | 2.9% | $2,916 | 2 | $1,088.00 | $578.21 | 3.5% → 6.0% | 93% | 3.27 | 2.51 |
| V | Financials | BUY | 5.8% | $5,805 | 15 | $372.10 | $324.61 | 1.8% → 2.0% | 86% | 0.30 | 0.11 |
| NUE | Materials | BUY | 5.8% | $5,794 | 23 | $246.44 | $200.56 | 3.3% → 4.0% | 96% | 0.95 | 1.64 |
| SPG | Real Estate | BUY | 5.7% | $5,718 | 28 | $197.59 | $176.16 | 2.0% → 2.1% | 91% | 0.26 | 0.48 |
| TMO | Health Care | BUY | 5.7% | $5,715 | 8 | $662.06 | $551.33 | 2.3% → 2.5% | 90% | 0.54 | 0.45 |
| VZ | Communication Services | BUY - half size | 2.8% | $2,818 | 61 | $45.77 | $39.18 | 0.9% → 0.8% | 90% | -0.29 | 0.78 |
| CRM | Information Technology | BUY - half size | 2.7% | $2,740 | 12 | $224.56 | $162.82 | 1.3% → 0.5% | 84% | 0.44 | -0.14 |
| PM | Consumer Staples | BUY | 5.3% | $5,257 | 27 | $192.69 | $161.73 | 1.3% → 1.4% | 92% | -0.16 | 1.04 |
| MRK | Health Care | BUY | 5.3% | $5,255 | 36 | $142.79 | $118.80 | 2.5% → 2.9% | 96% | 0.17 | 1.65 |

Cash / T-bills: 16.2%

## MPC - BUY

**Thesis.** MPC scores higher than 100% of the universe on the factor composite (12-1 momentum z = +2.60, 6-1 momentum z = +3.84, low-vol z = -0.41, trend z = +3.41). Black-Litterman blends the market-equilibrium return of 0.4% with that factor view to a posterior expected return of 1.7%/yr, and the mean-variance step gives it a 10.0% weight after accounting for its correlation with the other holdings.

**Statistical evidence**

- Empirical-Bayes drift: after shrinking its past-year mean return 87% toward the cross-section, the posterior probability its true drift is positive is 97%.
- Kalman-filter trend t-statistic: +2.10 (state-space estimate of the current drift ÷ its uncertainty).
- CAPM vs SPY (Newey-West errors): beta -0.13, alpha 88.7%/yr, t = 2.33, p = 0.020. Under a skeptical prior the posterior alpha is 0.2%/yr and P(alpha > 0) = 55% (Bayes factor 1.01).
- One-year Sharpe 2.45; probabilistic Sharpe P(true Sharpe > 0) = 99%. Bootstrap 90% CI for the annualized mean return: 33.7% to 148.1%.

**Risk**

- Volatility 35.4%/yr; max drawdown over the year -18.3%.
- 1-day 95% VaR 3.60% (historical) / 3.31% (Cornish-Fisher); expected shortfall 4.81%. On $10,000 that is a typical bad-day loss of $360.
- Average correlation with the other picks: 0.04.

**Decision gates**

  - PASS: BL posterior tilt > 0
  - PASS: EB P(drift>0) >= 80%
  - PASS: beta <= 1.6
  - PASS: avg corr to other picks <= 0.6

**Exit plan.** Sell at the monthly rebalance if it leaves the Black-Litterman top 15; sell immediately on a close below $351.74 (a two-sigma monthly move from $442.26).

**Caveat.** One year of one stock's data is too short to prove skill on its own (most single-stock alpha t-statistics are below 2); the case for owning it rests on the strategy-level evidence above, and these statistics confirm it is not an outlier on risk.

## PSX - BUY

**Thesis.** PSX scores higher than 99% of the universe on the factor composite (12-1 momentum z = +2.37, 6-1 momentum z = +2.88, low-vol z = +0.03, trend z = +3.41). Black-Litterman blends the market-equilibrium return of 0.5% with that factor view to a posterior expected return of 1.7%/yr, and the mean-variance step gives it a 10.0% weight after accounting for its correlation with the other holdings.

**Statistical evidence**

- Empirical-Bayes drift: after shrinking its past-year mean return 83% toward the cross-section, the posterior probability its true drift is positive is 98%.
- Kalman-filter trend t-statistic: +1.82 (state-space estimate of the current drift ÷ its uncertainty).
- CAPM vs SPY (Newey-West errors): beta -0.23, alpha 78.7%/yr, t = 2.44, p = 0.015. Under a skeptical prior the posterior alpha is 0.3%/yr and P(alpha > 0) = 56% (Bayes factor 1.01).
- One-year Sharpe 2.42; probabilistic Sharpe P(true Sharpe > 0) = 99%. Bootstrap 90% CI for the annualized mean return: 29.7% to 131.6%.

**Risk**

- Volatility 31.3%/yr; max drawdown over the year -17.3%.
- 1-day 95% VaR 2.85% (historical) / 2.84% (Cornish-Fisher); expected shortfall 3.89%. On $10,000 that is a typical bad-day loss of $285.
- Average correlation with the other picks: 0.06.

**Decision gates**

  - PASS: BL posterior tilt > 0
  - PASS: EB P(drift>0) >= 80%
  - PASS: beta <= 1.6
  - PASS: avg corr to other picks <= 0.6

**Exit plan.** Sell at the monthly rebalance if it leaves the Black-Litterman top 15; sell immediately on a close below $222.49 (a two-sigma monthly move from $271.62).

**Caveat.** One year of one stock's data is too short to prove skill on its own (most single-stock alpha t-statistics are below 2); the case for owning it rests on the strategy-level evidence above, and these statistics confirm it is not an outlier on risk.

## MSFT - BUY - half size

**Thesis.** MSFT scores higher than 77% of the universe on the factor composite (12-1 momentum z = -0.63, 6-1 momentum z = +1.48, low-vol z = -0.77, trend z = +1.62). Black-Litterman blends the market-equilibrium return of 1.5% with that factor view to a posterior expected return of 1.2%/yr, and the mean-variance step gives it a 4.2% weight after accounting for its correlation with the other holdings.

**Statistical evidence**

- Empirical-Bayes drift: after shrinking its past-year mean return 85% toward the cross-section, the posterior probability its true drift is positive is 85%.
- Kalman-filter trend t-statistic: +0.79 (state-space estimate of the current drift ÷ its uncertainty).
- CAPM vs SPY (Newey-West errors): beta 0.96, alpha -9.8%/yr, t = -0.32, p = 0.749. Under a skeptical prior the posterior alpha is -0.0%/yr and P(alpha > 0) = 49% (Bayes factor 1.00).
- One-year Sharpe 0.08; probabilistic Sharpe P(true Sharpe > 0) = 53%. Bootstrap 90% CI for the annualized mean return: -47.5% to 58.9%.

**Risk**

- Volatility 32.7%/yr; max drawdown over the year -34.5%.
- 1-day 95% VaR 2.80% (historical) / 1.94% (Cornish-Fisher); expected shortfall 4.05%. On $4,202 that is a typical bad-day loss of $118.
- Average correlation with the other picks: 0.00.

**Decision gates**

  - FAIL: BL posterior tilt > 0
  - PASS: EB P(drift>0) >= 80%
  - PASS: beta <= 1.6
  - PASS: avg corr to other picks <= 0.6

**Exit plan.** Sell at the monthly rebalance if it leaves the Black-Litterman top 15; sell immediately on a close below $429.85 (a two-sigma monthly move from $529.76).

**Caveat.** One year of one stock's data is too short to prove skill on its own (most single-stock alpha t-statistics are below 2); the case for owning it rests on the strategy-level evidence above, and these statistics confirm it is not an outlier on risk.

## KO - BUY

**Thesis.** KO scores higher than 90% of the universe on the factor composite (12-1 momentum z = +0.56, 6-1 momentum z = +0.46, low-vol z = +0.88, trend z = +0.52). Black-Litterman blends the market-equilibrium return of 0.9% with that factor view to a posterior expected return of 1.0%/yr, and the mean-variance step gives it a 8.0% weight after accounting for its correlation with the other holdings.

**Statistical evidence**

- Empirical-Bayes drift: after shrinking its past-year mean return 64% toward the cross-section, the posterior probability its true drift is positive is 96%.
- Kalman-filter trend t-statistic: +0.61 (state-space estimate of the current drift ÷ its uncertainty).
- CAPM vs SPY (Newey-West errors): beta -0.24, alpha 29.9%/yr, t = 1.63, p = 0.104. Under a skeptical prior the posterior alpha is 0.3%/yr and P(alpha > 0) = 57% (Bayes factor 1.01).
- One-year Sharpe 1.43; probabilistic Sharpe P(true Sharpe > 0) = 93%. Bootstrap 90% CI for the annualized mean return: -0.5% to 61.4%.

**Risk**

- Volatility 18.7%/yr; max drawdown over the year -7.9%.
- 1-day 95% VaR 1.56% (historical) / 1.54% (Cornish-Fisher); expected shortfall 2.01%. On $8,037 that is a typical bad-day loss of $125.
- Average correlation with the other picks: 0.08.

**Decision gates**

  - PASS: BL posterior tilt > 0
  - PASS: EB P(drift>0) >= 80%
  - PASS: beta <= 1.6
  - PASS: avg corr to other picks <= 0.6

**Exit plan.** Sell at the monthly rebalance if it leaves the Black-Litterman top 15; sell immediately on a close below $76.54 (a two-sigma monthly move from $85.82).

**Caveat.** One year of one stock's data is too short to prove skill on its own (most single-stock alpha t-statistics are below 2); the case for owning it rests on the strategy-level evidence above, and these statistics confirm it is not an outlier on risk.

## AMD - BUY - half size

**Thesis.** AMD scores higher than 98% of the universe on the factor composite (12-1 momentum z = +3.71, 6-1 momentum z = +3.85, low-vol z = -2.80, trend z = +3.41). Black-Litterman blends the market-equilibrium return of 4.1% with that factor view to a posterior expected return of 5.8%/yr, and the mean-variance step gives it a 3.5% weight after accounting for its correlation with the other holdings.

**Statistical evidence**

- Empirical-Bayes drift: after shrinking its past-year mean return 96% toward the cross-section, the posterior probability its true drift is positive is 92%.
- Kalman-filter trend t-statistic: +1.17 (state-space estimate of the current drift ÷ its uncertainty).
- CAPM vs SPY (Newey-West errors): beta 3.17, alpha 94.7%/yr, t = 1.79, p = 0.073. Under a skeptical prior the posterior alpha is 0.1%/yr and P(alpha > 0) = 53% (Bayes factor 1.00).
- One-year Sharpe 1.96; probabilistic Sharpe P(true Sharpe > 0) = 98%. Bootstrap 90% CI for the annualized mean return: 30.9% to 253.3%.

**Risk**

- Volatility 69.0%/yr; max drawdown over the year -27.8%.
- 1-day 95% VaR 5.72% (historical) / 6.03% (Cornish-Fisher); expected shortfall 8.18%. On $3,535 that is a typical bad-day loss of $202.
- Average correlation with the other picks: -0.05.

**Decision gates**

  - PASS: BL posterior tilt > 0
  - PASS: EB P(drift>0) >= 80%
  - FAIL: beta <= 1.6
  - PASS: avg corr to other picks <= 0.6

**Exit plan.** Sell at the monthly rebalance if it leaves the Black-Litterman top 15; sell immediately on a close below $388.61 (a two-sigma monthly move from $645.86).

**Caveat.** One year of one stock's data is too short to prove skill on its own (most single-stock alpha t-statistics are below 2); the case for owning it rests on the strategy-level evidence above, and these statistics confirm it is not an outlier on risk.

## JNJ - BUY

**Thesis.** JNJ scores higher than 92% of the universe on the factor composite (12-1 momentum z = +0.84, 6-1 momentum z = +0.33, low-vol z = +0.84, trend z = +0.48). Black-Litterman blends the market-equilibrium return of 1.1% with that factor view to a posterior expected return of 1.3%/yr, and the mean-variance step gives it a 6.0% weight after accounting for its correlation with the other holdings.

**Statistical evidence**

- Empirical-Bayes drift: after shrinking its past-year mean return 65% toward the cross-section, the posterior probability its true drift is positive is 97%.
- Kalman-filter trend t-statistic: +0.63 (state-space estimate of the current drift ÷ its uncertainty).
- CAPM vs SPY (Newey-West errors): beta -0.18, alpha 34.5%/yr, t = 1.66, p = 0.098. Under a skeptical prior the posterior alpha is 0.3%/yr and P(alpha > 0) = 56% (Bayes factor 1.01).
- One-year Sharpe 1.68; probabilistic Sharpe P(true Sharpe > 0) = 95%. Bootstrap 90% CI for the annualized mean return: 5.3% to 67.3%.

**Risk**

- Volatility 19.1%/yr; max drawdown over the year -11.0%.
- 1-day 95% VaR 1.78% (historical) / 1.78% (Cornish-Fisher); expected shortfall 2.35%. On $5,996 that is a typical bad-day loss of $107.
- Average correlation with the other picks: 0.08.

**Decision gates**

  - PASS: BL posterior tilt > 0
  - PASS: EB P(drift>0) >= 80%
  - PASS: beta <= 1.6
  - PASS: avg corr to other picks <= 0.6

**Exit plan.** Sell at the monthly rebalance if it leaves the Black-Litterman top 15; sell immediately on a close below $229.89 (a two-sigma monthly move from $258.45).

**Caveat.** One year of one stock's data is too short to prove skill on its own (most single-stock alpha t-statistics are below 2); the case for owning it rests on the strategy-level evidence above, and these statistics confirm it is not an outlier on risk.

## MU - BUY - half size

**Thesis.** MU scores higher than 98% of the universe on the factor composite (12-1 momentum z = +4.16, 6-1 momentum z = +3.85, low-vol z = -3.06, trend z = +3.41). Black-Litterman blends the market-equilibrium return of 3.5% with that factor view to a posterior expected return of 6.0%/yr, and the mean-variance step gives it a 2.9% weight after accounting for its correlation with the other holdings.

**Statistical evidence**

- Empirical-Bayes drift: after shrinking its past-year mean return 97% toward the cross-section, the posterior probability its true drift is positive is 93%.
- Kalman-filter trend t-statistic: +1.02 (state-space estimate of the current drift ÷ its uncertainty).
- CAPM vs SPY (Newey-West errors): beta 3.27, alpha 161.4%/yr, t = 2.51, p = 0.012. Under a skeptical prior the posterior alpha is 0.2%/yr and P(alpha > 0) = 53% (Bayes factor 1.00).
- One-year Sharpe 2.51; probabilistic Sharpe P(true Sharpe > 0) = 99%. Bootstrap 90% CI for the annualized mean return: 79.6% to 342.5%.

**Risk**

- Volatility 81.2%/yr; max drawdown over the year -39.1%.
- 1-day 95% VaR 7.00% (historical) / 7.00% (Cornish-Fisher); expected shortfall 9.53%. On $2,916 that is a typical bad-day loss of $204.
- Average correlation with the other picks: -0.05.

**Decision gates**

  - PASS: BL posterior tilt > 0
  - PASS: EB P(drift>0) >= 80%
  - FAIL: beta <= 1.6
  - PASS: avg corr to other picks <= 0.6

**Exit plan.** Sell at the monthly rebalance if it leaves the Black-Litterman top 15; sell immediately on a close below $578.21 (a two-sigma monthly move from $1,088.00).

**Caveat.** One year of one stock's data is too short to prove skill on its own (most single-stock alpha t-statistics are below 2); the case for owning it rests on the strategy-level evidence above, and these statistics confirm it is not an outlier on risk.

## V - BUY

**Thesis.** V scores higher than 85% of the universe on the factor composite (12-1 momentum z = -0.29, 6-1 momentum z = +0.74, low-vol z = +0.96, trend z = +0.70). Black-Litterman blends the market-equilibrium return of 1.8% with that factor view to a posterior expected return of 2.0%/yr, and the mean-variance step gives it a 5.8% weight after accounting for its correlation with the other holdings.

**Statistical evidence**

- Empirical-Bayes drift: after shrinking its past-year mean return 72% toward the cross-section, the posterior probability its true drift is positive is 86%.
- Kalman-filter trend t-statistic: +0.53 (state-space estimate of the current drift ÷ its uncertainty).
- CAPM vs SPY (Newey-West errors): beta 0.30, alpha 2.0%/yr, t = 0.11, p = 0.916. Under a skeptical prior the posterior alpha is 0.0%/yr and P(alpha > 0) = 50% (Bayes factor 0.99).
- One-year Sharpe 0.27; probabilistic Sharpe P(true Sharpe > 0) = 61%. Bootstrap 90% CI for the annualized mean return: -27.9% to 46.9%.

**Risk**

- Volatility 22.1%/yr; max drawdown over the year -17.2%.
- 1-day 95% VaR 1.80% (historical) / 1.70% (Cornish-Fisher); expected shortfall 2.77%. On $5,805 that is a typical bad-day loss of $105.
- Average correlation with the other picks: 0.09.

**Decision gates**

  - PASS: BL posterior tilt > 0
  - PASS: EB P(drift>0) >= 80%
  - PASS: beta <= 1.6
  - PASS: avg corr to other picks <= 0.6

**Exit plan.** Sell at the monthly rebalance if it leaves the Black-Litterman top 15; sell immediately on a close below $324.61 (a two-sigma monthly move from $372.10).

**Caveat.** One year of one stock's data is too short to prove skill on its own (most single-stock alpha t-statistics are below 2); the case for owning it rests on the strategy-level evidence above, and these statistics confirm it is not an outlier on risk.

## NUE - BUY

**Thesis.** NUE scores higher than 97% of the universe on the factor composite (12-1 momentum z = +2.09, 6-1 momentum z = +1.99, low-vol z = -0.51, trend z = +1.02). Black-Litterman blends the market-equilibrium return of 3.3% with that factor view to a posterior expected return of 4.0%/yr, and the mean-variance step gives it a 5.8% weight after accounting for its correlation with the other holdings.

**Statistical evidence**

- Empirical-Bayes drift: after shrinking its past-year mean return 84% toward the cross-section, the posterior probability its true drift is positive is 96%.
- Kalman-filter trend t-statistic: +0.69 (state-space estimate of the current drift ÷ its uncertainty).
- CAPM vs SPY (Newey-West errors): beta 0.95, alpha 50.3%/yr, t = 1.64, p = 0.101. Under a skeptical prior the posterior alpha is 0.2%/yr and P(alpha > 0) = 54% (Bayes factor 1.00).
- One-year Sharpe 1.94; probabilistic Sharpe P(true Sharpe > 0) = 97%. Bootstrap 90% CI for the annualized mean return: 8.9% to 120.4%.

**Risk**

- Volatility 32.2%/yr; max drawdown over the year -18.4%.
- 1-day 95% VaR 3.02% (historical) / 3.05% (Cornish-Fisher); expected shortfall 3.88%. On $5,794 that is a typical bad-day loss of $175.
- Average correlation with the other picks: 0.08.

**Decision gates**

  - PASS: BL posterior tilt > 0
  - PASS: EB P(drift>0) >= 80%
  - PASS: beta <= 1.6
  - PASS: avg corr to other picks <= 0.6

**Exit plan.** Sell at the monthly rebalance if it leaves the Black-Litterman top 15; sell immediately on a close below $200.56 (a two-sigma monthly move from $246.44).

**Caveat.** One year of one stock's data is too short to prove skill on its own (most single-stock alpha t-statistics are below 2); the case for owning it rests on the strategy-level evidence above, and these statistics confirm it is not an outlier on risk.

## SPG - BUY

**Thesis.** SPG scores higher than 73% of the universe on the factor composite (12-1 momentum z = +0.17, 6-1 momentum z = +0.19, low-vol z = +1.23, trend z = -0.20). Black-Litterman blends the market-equilibrium return of 2.0% with that factor view to a posterior expected return of 2.1%/yr, and the mean-variance step gives it a 5.7% weight after accounting for its correlation with the other holdings.

**Statistical evidence**

- Empirical-Bayes drift: after shrinking its past-year mean return 64% toward the cross-section, the posterior probability its true drift is positive is 91%.
- Kalman-filter trend t-statistic: -0.13 (state-space estimate of the current drift ÷ its uncertainty).
- CAPM vs SPY (Newey-West errors): beta 0.26, alpha 9.1%/yr, t = 0.48, p = 0.629. Under a skeptical prior the posterior alpha is 0.1%/yr and P(alpha > 0) = 52% (Bayes factor 1.00).
- One-year Sharpe 0.66; probabilistic Sharpe P(true Sharpe > 0) = 75%. Bootstrap 90% CI for the annualized mean return: -13.4% to 46.2%.

**Risk**

- Volatility 18.8%/yr; max drawdown over the year -15.6%.
- 1-day 95% VaR 1.74% (historical) / 1.83% (Cornish-Fisher); expected shortfall 2.40%. On $5,718 that is a typical bad-day loss of $100.
- Average correlation with the other picks: 0.09.

**Decision gates**

  - PASS: BL posterior tilt > 0
  - PASS: EB P(drift>0) >= 80%
  - PASS: beta <= 1.6
  - PASS: avg corr to other picks <= 0.6

**Exit plan.** Sell at the monthly rebalance if it leaves the Black-Litterman top 15; sell immediately on a close below $176.16 (a two-sigma monthly move from $197.59).

**Caveat.** One year of one stock's data is too short to prove skill on its own (most single-stock alpha t-statistics are below 2); the case for owning it rests on the strategy-level evidence above, and these statistics confirm it is not an outlier on risk.

## TMO - BUY

**Thesis.** TMO scores higher than 86% of the universe on the factor composite (12-1 momentum z = -0.15, 6-1 momentum z = +0.75, low-vol z = +0.05, trend z = +1.57). Black-Litterman blends the market-equilibrium return of 2.3% with that factor view to a posterior expected return of 2.5%/yr, and the mean-variance step gives it a 5.7% weight after accounting for its correlation with the other holdings.

**Statistical evidence**

- Empirical-Bayes drift: after shrinking its past-year mean return 81% toward the cross-section, the posterior probability its true drift is positive is 90%.
- Kalman-filter trend t-statistic: +0.96 (state-space estimate of the current drift ÷ its uncertainty).
- CAPM vs SPY (Newey-West errors): beta 0.54, alpha 13.5%/yr, t = 0.45, p = 0.656. Under a skeptical prior the posterior alpha is 0.1%/yr and P(alpha > 0) = 51% (Bayes factor 1.00).
- One-year Sharpe 0.71; probabilistic Sharpe P(true Sharpe > 0) = 76%. Bootstrap 90% CI for the annualized mean return: -23.8% to 70.3%.

**Risk**

- Volatility 29.0%/yr; max drawdown over the year -31.4%.
- 1-day 95% VaR 2.57% (historical) / 2.65% (Cornish-Fisher); expected shortfall 3.62%. On $5,715 that is a typical bad-day loss of $147.
- Average correlation with the other picks: 0.06.

**Decision gates**

  - PASS: BL posterior tilt > 0
  - PASS: EB P(drift>0) >= 80%
  - PASS: beta <= 1.6
  - PASS: avg corr to other picks <= 0.6

**Exit plan.** Sell at the monthly rebalance if it leaves the Black-Litterman top 15; sell immediately on a close below $551.33 (a two-sigma monthly move from $662.06).

**Caveat.** One year of one stock's data is too short to prove skill on its own (most single-stock alpha t-statistics are below 2); the case for owning it rests on the strategy-level evidence above, and these statistics confirm it is not an outlier on risk.

## VZ - BUY - half size

**Thesis.** VZ scores higher than 68% of the universe on the factor composite (12-1 momentum z = +0.38, 6-1 momentum z = +0.07, low-vol z = +0.60, trend z = -0.07). Black-Litterman blends the market-equilibrium return of 0.9% with that factor view to a posterior expected return of 0.8%/yr, and the mean-variance step gives it a 2.8% weight after accounting for its correlation with the other holdings.

**Statistical evidence**

- Empirical-Bayes drift: after shrinking its past-year mean return 76% toward the cross-section, the posterior probability its true drift is positive is 90%.
- Kalman-filter trend t-statistic: +0.07 (state-space estimate of the current drift ÷ its uncertainty).
- CAPM vs SPY (Newey-West errors): beta -0.29, alpha 19.6%/yr, t = 0.78, p = 0.437. Under a skeptical prior the posterior alpha is 0.1%/yr and P(alpha > 0) = 52% (Bayes factor 1.00).
- One-year Sharpe 0.64; probabilistic Sharpe P(true Sharpe > 0) = 74%. Bootstrap 90% CI for the annualized mean return: -18.7% to 58.6%.

**Risk**

- Volatility 24.9%/yr; max drawdown over the year -17.0%.
- 1-day 95% VaR 2.22% (historical) / 1.32% (Cornish-Fisher); expected shortfall 3.14%. On $2,818 that is a typical bad-day loss of $62.
- Average correlation with the other picks: 0.05.

**Decision gates**

  - FAIL: BL posterior tilt > 0
  - PASS: EB P(drift>0) >= 80%
  - PASS: beta <= 1.6
  - PASS: avg corr to other picks <= 0.6

**Exit plan.** Sell at the monthly rebalance if it leaves the Black-Litterman top 15; sell immediately on a close below $39.18 (a two-sigma monthly move from $45.77).

**Caveat.** One year of one stock's data is too short to prove skill on its own (most single-stock alpha t-statistics are below 2); the case for owning it rests on the strategy-level evidence above, and these statistics confirm it is not an outlier on risk.

## CRM - BUY - half size

**Thesis.** CRM scores higher than 48% of the universe on the factor composite (12-1 momentum z = -0.41, 6-1 momentum z = +2.04, low-vol z = -2.58, trend z = +0.87). Black-Litterman blends the market-equilibrium return of 1.3% with that factor view to a posterior expected return of 0.5%/yr, and the mean-variance step gives it a 2.7% weight after accounting for its correlation with the other holdings.

**Statistical evidence**

- Empirical-Bayes drift: after shrinking its past-year mean return 92% toward the cross-section, the posterior probability its true drift is positive is 84%.
- Kalman-filter trend t-statistic: +0.32 (state-space estimate of the current drift ÷ its uncertainty).
- CAPM vs SPY (Newey-West errors): beta 0.44, alpha -6.6%/yr, t = -0.14, p = 0.889. Under a skeptical prior the posterior alpha is -0.0%/yr and P(alpha > 0) = 50% (Bayes factor 1.00).
- One-year Sharpe -0.02; probabilistic Sharpe P(true Sharpe > 0) = 49%. Bootstrap 90% CI for the annualized mean return: -77.4% to 79.2%.

**Risk**

- Volatility 47.6%/yr; max drawdown over the year -43.3%.
- 1-day 95% VaR 4.10% (historical) / 2.53% (Cornish-Fisher); expected shortfall 5.47%. On $2,740 that is a typical bad-day loss of $112.
- Average correlation with the other picks: 0.01.

**Decision gates**

  - FAIL: BL posterior tilt > 0
  - PASS: EB P(drift>0) >= 80%
  - PASS: beta <= 1.6
  - PASS: avg corr to other picks <= 0.6

**Exit plan.** Sell at the monthly rebalance if it leaves the Black-Litterman top 15; sell immediately on a close below $162.82 (a two-sigma monthly move from $224.56).

**Caveat.** One year of one stock's data is too short to prove skill on its own (most single-stock alpha t-statistics are below 2); the case for owning it rests on the strategy-level evidence above, and these statistics confirm it is not an outlier on risk.

## PM - BUY

**Thesis.** PM scores higher than 79% of the universe on the factor composite (12-1 momentum z = +0.20, 6-1 momentum z = +0.50, low-vol z = +0.42, trend z = +0.65). Black-Litterman blends the market-equilibrium return of 1.3% with that factor view to a posterior expected return of 1.4%/yr, and the mean-variance step gives it a 5.3% weight after accounting for its correlation with the other holdings.

**Statistical evidence**

- Empirical-Bayes drift: after shrinking its past-year mean return 80% toward the cross-section, the posterior probability its true drift is positive is 92%.
- Kalman-filter trend t-statistic: +0.53 (state-space estimate of the current drift ÷ its uncertainty).
- CAPM vs SPY (Newey-West errors): beta -0.16, alpha 28.5%/yr, t = 1.04, p = 0.297. Under a skeptical prior the posterior alpha is 0.2%/yr and P(alpha > 0) = 53% (Bayes factor 1.00).
- One-year Sharpe 0.95; probabilistic Sharpe P(true Sharpe > 0) = 83%. Bootstrap 90% CI for the annualized mean return: -15.5% to 75.4%.

**Risk**

- Volatility 27.8%/yr; max drawdown over the year -18.5%.
- 1-day 95% VaR 2.94% (historical) / 2.67% (Cornish-Fisher); expected shortfall 3.65%. On $5,257 that is a typical bad-day loss of $154.
- Average correlation with the other picks: 0.05.

**Decision gates**

  - PASS: BL posterior tilt > 0
  - PASS: EB P(drift>0) >= 80%
  - PASS: beta <= 1.6
  - PASS: avg corr to other picks <= 0.6

**Exit plan.** Sell at the monthly rebalance if it leaves the Black-Litterman top 15; sell immediately on a close below $161.73 (a two-sigma monthly move from $192.69).

**Caveat.** One year of one stock's data is too short to prove skill on its own (most single-stock alpha t-statistics are below 2); the case for owning it rests on the strategy-level evidence above, and these statistics confirm it is not an outlier on risk.

## MRK - BUY

**Thesis.** MRK scores higher than 95% of the universe on the factor composite (12-1 momentum z = +1.54, 6-1 momentum z = +0.83, low-vol z = -0.37, trend z = +1.20). Black-Litterman blends the market-equilibrium return of 2.5% with that factor view to a posterior expected return of 2.9%/yr, and the mean-variance step gives it a 5.3% weight after accounting for its correlation with the other holdings.

**Statistical evidence**

- Empirical-Bayes drift: after shrinking its past-year mean return 81% toward the cross-section, the posterior probability its true drift is positive is 96%.
- Kalman-filter trend t-statistic: +0.88 (state-space estimate of the current drift ÷ its uncertainty).
- CAPM vs SPY (Newey-West errors): beta 0.17, alpha 48.7%/yr, t = 1.65, p = 0.098. Under a skeptical prior the posterior alpha is 0.2%/yr and P(alpha > 0) = 54% (Bayes factor 1.00).
- One-year Sharpe 1.75; probabilistic Sharpe P(true Sharpe > 0) = 97%. Bootstrap 90% CI for the annualized mean return: 9.6% to 103.8%.

**Risk**

- Volatility 29.1%/yr; max drawdown over the year -11.4%.
- 1-day 95% VaR 2.40% (historical) / 1.69% (Cornish-Fisher); expected shortfall 3.05%. On $5,255 that is a typical bad-day loss of $126.
- Average correlation with the other picks: 0.11.

**Decision gates**

  - PASS: BL posterior tilt > 0
  - PASS: EB P(drift>0) >= 80%
  - PASS: beta <= 1.6
  - PASS: avg corr to other picks <= 0.6

**Exit plan.** Sell at the monthly rebalance if it leaves the Black-Litterman top 15; sell immediately on a close below $118.80 (a two-sigma monthly move from $142.79).

**Caveat.** One year of one stock's data is too short to prove skill on its own (most single-stock alpha t-statistics are below 2); the case for owning it rests on the strategy-level evidence above, and these statistics confirm it is not an outlier on risk.
