# Group Z – Next steps: how the strategy could still be improved

23 Nov 2026 · Felix Schoenstein · follow-up to `GroupZ_Notebook_Update_Handover.md`

## Starting point

The extended tests (notebook §9.6–9.13) showed that more ML does not improve the strategy: 14 variants, none passes its gates, classifier AUC ≈ 0.5. The remaining room for improvement lies in **measured weaknesses of the rule itself**, not in better forecasts:

| Weakness | Evidence from the tests | Approach |
| --- | --- | --- |
| Result depends on the rebalancing day | Sharpe 0.55–0.66 and max drawdown −5.7% to −8.9% across rebalancing 0–15 trading days before month-end; month-end was the best of four | 1. Tranche rebalancing |
| High turnover, thin cost margin | Turnover 216% p.a.; break-even cost only 8 bps; Sharpe 0.66 → 0.47 at 50 bps | 2. Signal hysteresis and no-trade bands |
| Late re-entry after quick rebounds | 2020: +4.8% vs +14.6% for 60/40 | 2. Hysteresis (partly), 3. averaged signal |
| Promising but unproven variant | Averaged 6–12-month trend signal (H): Sharpe 0.71 vs 0.66, higher in both halves, but fails the gates | 3. Out-of-sample test on longer history |
| Interest-rate concentration | Treasuries are the largest holding (33%); 2022 loss −3.4% came from bonds and equities falling together | 4a. Inflation-linked bonds |
| Low absolute return | 4.6% vs 7.9% p.a. for 60/40; leverage lowered the Sharpe ratio at an assumed 0.5% borrowing spread | 4b. Cheaper leverage (sensitivity) |

**Recommended order:** 1 → 2 → 3. Items 1 and 2 target real, measured problems, are easy to explain to cautious clients and take little code. Item 3 is the clean way to decide on variant H. Item 4 only if time remains.

## General protocol (applies to every approach)

The extended tests are credible because they were pre-registered. Keep it that way:

1. **Fix everything before the first run** in a frozen dataclass (e.g. `ImproveConfig`): parameters, thresholds, data sources, gates.
2. **One run, report everything**, including failures. Record each test in the observation ledger (§9.5 / §9.13).
3. **Same gates as §9.2** (Sharpe ≥ A, max drawdown not deeper than A, better than A in at least 2 of 3 stress periods), plus the approach-specific gate named below.
4. **Count the variants.** We have already tested 17 (variants B–D in §9, 14 in §9.6–9.13). Every further one raises the chance that something wins by luck. Optional: report the deflated Sharpe ratio (Bailey and López de Prado, 2014), which corrects for the number of trials.
5. **Never tune parameters to the backtest.** Sensitivity grids may be shown (like Figure 4), but the tested value is the one fixed in advance.

## 1. Tranche rebalancing (priority 1)

**Idea.** Split the portfolio into four equal tranches that rebalance on different days: the last trading day of the month, and 5, 10 and 15 trading days before it. Each tranche follows the unchanged rule. The combined portfolio averages out the timing luck instead of depending on one day. This is the only approach that makes the **strategy itself** more robust, rather than only the evidence for it.

**Implementation.**

- `market_data_offset(prices_raw, k)` in §9.12 already builds the month-end data for each offset k.
- New: a daily backtest engine `backtest_daily(weights, data, cost_bps)` that holds each tranche's weights from its decision day to the next, lets them drift with daily prices, charges costs on turnover and earns the daily T-bill rate on cash. Check first: for k = 0 its month-end NAV must reproduce variant A (assert, tolerance about 1e-6 per month).
- Combined NAV = average of the four tranche NAVs, each starting with 25% of capital and never re-balanced against each other. For the performance table, sample the combined NAV at month-ends, so all measures stay comparable with Table 2.

**Pre-registered settings.** Offsets (0, 5, 10, 15) trading days; equal tranche weights; 10 bps costs.

**Evaluation.** The usual gates against A, plus: "Sharpe and drawdown inside the range of the four single-offset runs". The aim is not a higher Sharpe; it is a result that no longer depends on one rebalancing day. A good outcome would be a Sharpe close to the average of the four offsets (about 0.61) with a lower spread.

**Expected effect.** Similar return, slightly smoother path, same total turnover (each tranche trades a quarter of the size). Effort: about 4–6 h (the daily engine is the main work).

## 2. Lower turnover: signal hysteresis and no-trade bands (priority 2)

**Idea A – hysteresis on the trend signal.** Today an asset is sold as soon as its price closes below its 10-month SMA and bought back as soon as it closes above. Near the average this flips back and forth. With a band b:

- held asset: sell only if price < SMA × (1 − b);
- asset in cash: buy back only if price > SMA × (1 + b);
- otherwise keep the previous state.

This is a small state machine per asset in a new `trend_signal_hysteresis(prices_m, window, band)`. It uses only data up to t.

**Idea B – no-trade bands on the weights.** Do not trade a weight change smaller than a threshold c. Implement it at execution in a variant of `backtest`: `w_exec = where(|w_target − w_drifted| < c, w_drifted, w_target)`, with the difference going to or coming from cash. Trend switches are larger than c, so they are always executed; only the small risk-parity adjustments are skipped.

**Pre-registered settings.** b = 2% and c = 2 percentage points, chosen from common practice before the run. Optionally show a sensitivity grid (b = 1–3%, c = 1–3 pp) as a figure, like Figure 4, without picking from it.

**Evaluation.** Variants: hysteresis alone, bands alone, both. Gates against A, plus the approach-specific gate **"turnover at least 25% lower than A"**. Also report the break-even cost against 60/40 (fsolve, §8.1) and the Sharpe ratio at 20 and 50 bps.

**Expected effect.** Clearly lower turnover and a higher break-even cost; possibly fewer whipsaws such as in 2020. Risk: slightly later exits in a crash (a 2% band delays the sell signal). Effort: about 3–4 h.

## 3. Out-of-sample test of the averaged trend signal (priority 3)

**Idea.** Variant H looked promising but failed the gates, and we cannot loosen the gates afterwards. The clean decision is a **true out-of-sample test** on data the project has never used: the years before the ETFs existed.

**Data (team decision needed; check availability before pre-registering).** Long-history proxies for the six asset classes, for example:

- index mutual funds with long price histories on Yahoo Finance (e.g. Vanguard's S&P 500, intermediate-Treasury and REIT index funds);
- index series for gold and broad commodities (e.g. from FRED or the course's data access);
- or Global Financial Data / CRSP via WRDS, if Nova provides access.

Faber (2007) tests the same type of rule from 1973, which shows that long proxy histories are feasible. Developed ex-US equities and commodities are usually the hardest series to obtain.

**Test design.** Run only A and H on the proxy period (e.g. 1990–2006), with the existing gates and no other variants. One run.

**Decision rule (fix in advance).** If H passes the gates out of sample, it becomes the new prospectus strategy, and the prospectus says why. If not, A stays and H is reported as rejected.

Effort: 4–8 h, mostly data collection and checking.

## 4. Smaller ideas (only if time remains)

**4a. Inflation-linked Treasuries (TIP).** Add TIP (iShares TIPS Bond ETF, since Dec 2003, so the backtest start does not change) as a seventh asset class. Economic reason: TIPS respond to inflation surprises, while nominal Treasuries suffer, and inflation was the source of the 2022 joint fall of bonds and equities. Report: Sharpe, drawdown, the 2022 return and the change in the Treasury weight. Caveat to state honestly: the asset is added after seeing 2022, which is a form of hindsight. The economic argument should come first in the text.

**4b. Cheaper leverage – sensitivity, not a new strategy.** The levered variants used T-bills + 0.5%. With Treasury futures the implicit financing cost is closer to T-bills + 0.1–0.2%. Rerun the 10%-vol variant for spreads of 0, 0.1, 0.25 and 0.5%, and with a lower leverage cap of 2× (more realistic for private clients). Show the result as a sensitivity table for the client discussion in §6, not as a new strategy.

**4c. EUR investors (implementation, not performance).** The prospectus already recommends EUR-hedged bond and cash sleeves. A backtest in EUR with hedged and unhedged sleeves would quantify that recommendation. It needs EUR/USD and EUR rate data.

## What we would not do

- **More ML models or feature variants.** The classifiers reached AUC 0.49–0.52; further variants mostly add data-mining risk.
- **Picking the best cell of a parameter grid** (e.g. the 7-month SMA with Sharpe 0.81 in Figure 4). It is the classic in-sample trap.
- **Switching to individual stocks.** That would be a different product for different clients, with survivorship-biased free data.
- **Loosening the gates for variant H** after seeing its result.

## Overview

| # | Approach | Targets | Effort | Priority |
| --- | --- | --- | --- | --- |
| 1 | Tranche rebalancing (4 offsets) | timing luck | 4–6 h | Must |
| 2 | Hysteresis b = 2% and no-trade bands c = 2 pp | turnover, whipsaws | 3–4 h | Must |
| 3 | Out-of-sample test of H on 1990–2006 proxies | decision on H | 4–8 h | Should |
| 4a | TIP as 7th asset class | rate concentration | 2 h | Could |
| 4b | Leverage financing sensitivity | absolute return | 1 h | Could |
| 4c | EUR backtest with hedged sleeves | EUR clients | 3–4 h | Could |

## Open questions

- Do we still have time before the deadline for items 1 and 2?
- Item 3: which long-history data can we access (Yahoo mutual funds, FRED, WRDS)?
- If tranche rebalancing (item 1) passes: do we change the prospectus strategy, or report it as an implementation option for the wealth manager?

## References

- Bailey, D. H. and López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management*, 40(5), 94–107.
- Faber, M. T. (2007). A quantitative approach to tactical asset allocation. *Journal of Wealth Management*, 9(4), 69–79. (already cited)
