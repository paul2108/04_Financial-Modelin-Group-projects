# Group Z – Notebook update: extended ML section and robustness tests

23 Nov 2026 · Felix Schoenstein · for the colleague who maintains the notebook and the prospectus

## Bottom line

We tested five further ML model families and four robustness changes on the same data, months and costs as the prospectus. **No variant passes its pre-registered gates, so the prospectus strategy (risk parity + 10-month trend filter) stays unchanged.** What we gain is a much larger ML section for the grade and much stronger evidence that the base strategy's results are not luck.

The updated notebook is `GroupZ_RiskParity_Trend.ipynb` (sent together with this file). Sections 1–8 and the strategy itself are untouched.

## What changed in the notebook

| Where | Change |
| --- | --- |
| New cell after the title | Version note listing the changes below |
| Title cell | Table of contents row for Sections 9.6–9.13; runtime note (15–25 minutes) |
| §9.1 (cell with `register("ml_riskoff_hmm", ...)`) | Alarm shares of the HMM and the volatility rule now use the same 236 months as recall and precision (`DECISION_DATES[:-1]`). Before: 237 months |
| §9.2 (Table 3 cell) | New row "Stress periods better than A" in Table 3, so readers can see which gate each variant failed |
| §9.3 (error-analysis cell) | New register entry `ml_base_rate` (share of months followed by a stress month, about 10%) |
| **New §9.6–9.13** (24 cells, tagged `ext`, inserted before Section 10) | Extended ML models and robustness tests, see below. Placed before Section 10 so that their numbers go through `register(...)` into `results_macros.tex` |

**How to run.** Colab: *Runtime → Run all* (15–25 minutes; a GPU runtime makes FinBERT faster but is not needed). Locally: `make notebook` as before. The new sections need internet access to federalreserve.gov and Hugging Face. Outputs: all figures and tables as before, plus `groupz_test_results.txt` (all tables of §9.6–9.13) and `fomc_scores.csv` (FinBERT score per FOMC statement). On Colab both download automatically.

**Packages.** Colab already has all of them. For the local `.venv`, add to `requirements.txt` (pin the versions you install): `transformers`, `torch`, `beautifulsoup4`, `requests`. scikit-learn and SciPy are already used.

**Not changed:** the Streamlit app (none of the new variants was promoted, so the app needs no update; HRP could be added to the weighting dropdown later as an option).

## The new ML section (§9.6–9.13)

All models follow the design of the existing §9: walk-forward re-estimation once a year on an expanding window, only data up to each decision date, all settings fixed in a frozen `TestConfig` dataclass **before** the first run, and the same promotion gates as §9.2 (Sharpe ≥ A, max drawdown not deeper than A, better than A in at least 2 of 3 stress periods).

| Section | Model | ML family | What it does |
| --- | --- | --- | --- |
| 9.7 | Hierarchical Risk Parity (López de Prado, 2016) | unsupervised: hierarchical clustering | clusters assets by correlation and splits risk down the tree; alternative to the ERC weights |
| 9.7 | Ledoit-Wolf covariance | statistical learning: shrinkage | less noisy covariance matrix for the ERC weights |
| 9.7 | Averaged trend signal | ensemble of rules | holds each asset in proportion to the share of its 6–12-month SMAs it is above |
| 9.8 | Ex-ante volatility targeting | – | levers the strategy to 7% and 10% volatility (cap 4×, borrowing at T-bills + 0.5%) for a comparison with 60/40 at equal risk |
| 9.9 | 3-state Gaussian HMM | unsupervised: latent states | filtered regime probabilities as **features** (states sorted by volatility after every refit, so labels cannot swap) |
| 9.10 | FinBERT (`ProsusAI/finbert`) on FOMC statements | NLP / deep learning (transformer) | scores each sentence of every FOMC statement 2006–Oct 2026; statement tone = share positive − share negative |
| 9.11 | Logistic regression and gradient boosting | supervised classification | predict per asset whether next month's return is below T-bills, from HMM probabilities, trend distance, momentum, volatility, yield-curve slope, VIX, credit spread and FOMC tone |
| 9.12 | Circular block bootstrap, rebalance-date test, subsamples | resampling / robustness | confidence for the headline claims |

Classifier variants: **F** = classifier replaces the trend filter; **G** = a trend exit is followed only if the classifier also predicts a loss (meta-labelling). Ablations without the HMM and with the FOMC tone show what each feature group adds. Feature importance = drop in out-of-sample AUC when a feature group is shuffled.

Look-ahead safeguards worth mentioning in the prospectus: macro data are downloaded separately (the data fingerprint of Sections 1–8 is unchanged); the FOMC release date is read from each statement page, not from its web address (the 28 Jun 2007 statement sits at `.../20070618a.htm`); statements after the sample end are dropped.

## Results (run of 23 Nov 2026, Mar 2007–Oct 2026, net of 10 bps)

The run reproduced the prospectus exactly (A: Sharpe 0.66, max drawdown −5.65%, CAGR 4.6%).

| Variant | CAGR | Vol. | Sharpe | Max DD | Gates |
| --- | --- | --- | --- | --- | --- |
| A: Trend (base) | 4.6% | 4.7% | 0.66 | −5.7% | – |
| 60/40 | 7.9% | 9.9% | 0.67 | −32.3% | – |
| E: HRP + trend | 3.6% | 4.7% | 0.45 | −6.2% | fail |
| H: ERC + averaged trend | 4.8% | 4.6% | 0.71 | −6.3% | fail (drawdown, stress) |
| I: ERC (Ledoit-Wolf) + trend | 4.8% | 4.9% | 0.68 | −5.7% | fail (0 of 3 stress wins) |
| A levered to 7% vol | 5.6% | 6.7% | 0.62 | −12.5% | Sharpe below 60/40 |
| A levered to 10% vol | 7.0% | 9.6% | 0.60 | −17.4% | Sharpe below 60/40 |

Classifiers (evaluated Jan 2010–Oct 2026 because of the training burn-in; A has a Sharpe ratio of 0.65 over these months):

| Variant | Sharpe | Max DD | Gates |
| --- | --- | --- | --- |
| F: classifier replaces trend (logit / boosting) | 0.44 / 0.51 | −14.8% / −15.7% | fail |
| G: classifier filters trend exits (logit / boosting) | 0.57 / 0.59 | −14.5% / −15.9% | fail |
| G without HMM (logit / boosting) | 0.59 / 0.62 | −16.0% / −17.1% | fail |
| G with HMM + FOMC tone (logit / boosting) | 0.54 / 0.55 | −15.5% / −15.4% | fail |

Out-of-sample AUC of all classifiers: 0.49–0.52 (0.50 = coin flip). Within the models, the HMM probabilities are among the more useful inputs (shuffling them lowers AUC by 0.008–0.023), but models with and without them score almost the same (0.517 vs 0.513 for logit, 0.499 vs 0.503 for boosting). The FinBERT tone adds nothing (shuffling it changes AUC by less than 0.01).

**Interpretation.**

- **The classifiers cannot predict which asset will lose next month.** Variant G overrode about 60% of the trend filter's exits (238–278 of 391), which removed the crash protection and roughly tripled the drawdown.
- **HRP puts 45% in Treasuries** (ERC: 33%), which lowers the return and the Sharpe ratio.
- **The averaged trend signal (H) is the closest candidate.** Its Sharpe ratio is higher in both halves of the sample (0.70 vs 0.66 and 0.73 vs 0.65), and it beats A in 94% of bootstrap draws. But its drawdown is 0.6 pp deeper, and in two of the three stress periods it is equal to A (to one decimal), so it wins only 1 of 3. Because the gates were fixed in advance, it is not promoted. Report it as a candidate for a future out-of-sample test.
- **Leverage does not raise the Sharpe ratio**, because borrowing costs and the 4× cap bind in calm periods. At 60/40's volatility (9.6% vs 9.9%), though, the levered strategy earns 7.0% vs 7.9% a year with about half the drawdown (−17.4% vs −32.3%). This is the fairest way to show clients the trade-off.

## Why the strategy is now more robust

The strategy rule itself is unchanged. What is more robust is the **evidence** behind its claims, because every headline result was now attacked from several sides and survived:

1. **Drawdown advantage is near-certain, not a one-sample result.** In a 12-month block bootstrap (5,000 draws) the strategy's maximum drawdown is smaller than 60/40's in 99.9% of draws.
2. **"Same Sharpe as 60/40" is confirmed.** The strategy has the higher Sharpe ratio in 45.5% of draws (median difference −0.03), so the two are statistically indistinguishable, as the prospectus says.
3. **Stable over time.** Sharpe ratio 0.66 in Mar 2007–Dec 2016 and 0.65 in Jan 2017–Oct 2026 (60/40: 0.61 and 0.72).
4. **Not dependent on the covariance estimator.** Ledoit-Wolf shrinkage gives practically the same result (Sharpe 0.68, same drawdown).
5. **Not dependent on one trend length.** Averaging 6–12-month signals gives Sharpe 0.71, consistent with the plateau in Figure 4.
6. **Beats five ML alternatives.** Five ML model families, judged by gates fixed before the first run, did not beat the simple rule. That is evidence against data mining: the rule was not chosen because it won a horse race we tuned.
7. **Honest limitation (new).** Month-end happens to be the most favourable rebalancing day of the four tested. Rebalancing 5–15 trading days earlier gives Sharpe 0.55–0.62 and max drawdowns of −6.3% to −8.9%, still far below 60/40's −30% to −33%. The prospectus should state this range.

## Prospectus updates (suggested)

**§9: replace the last two sentences of "Result" and add a paragraph.** Suggested text:

> No variant passes the gates; the Sharpe differences of a few hundredths are also well within the statistical noise of §6 (standard error about 0.25). We then tested five further model families (§9.6–9.13 of the notebook): Hierarchical Risk Parity, a three-state HMM used as a feature in walk-forward logistic-regression and gradient-boosting classifiers, and FinBERT, a transformer language model that scores the tone of every FOMC statement since 2006. The classifiers reach an out-of-sample AUC of only 0.49–0.52, so they cannot tell which asset will fall next month, and overriding the trend filter roughly triples the drawdown. None of the 14 variants passes its gates. For our client this is a useful negative result: better prediction is not automatically a better portfolio, so the strategy stays rule-based.

**§8 (Risks and limitations): add to "Data snooping" or a new bullet "Statistical confidence":**

> In a block bootstrap of the monthly returns the strategy's maximum drawdown is smaller than that of 60/40 in 99.9% of 5,000 resampled histories, while its Sharpe ratio is higher in only 45.5%: the lower drawdown is robust, the risk-adjusted return is the same. Rebalancing 5–15 trading days before month-end gives Sharpe ratios of 0.55–0.62 and drawdowns of up to −8.9%.

**§6 (optional), one sentence on leverage:**

> Levered to the volatility of 60/40 (9.6%), the strategy would have earned 7.0% a year against 7.9% with a maximum drawdown of −17.4% against −32.3%, before taxes and with borrowing at the T-bill rate plus 0.5%.

**Table 3:** has the new stress-period row automatically. Optionally add a compact Table 4 with the 14 extended variants (from `groupz_test_results.txt`, section 9.13). Keep the ML part to about one page; details stay in the notebook.

**Numbers:** use the new register keys, never type them in by hand. The main ones: `ext_boot_p_mdd`, `ext_boot_p_sharpe`, `ext_auc_min`, `ext_auc_max`, `ext_n_variants`, `ext_n_promoted`, `ext_h_sharpe`, `ext_h_maxdd`, `ext_e_sharpe`, `ext_lev10_cagr`, `ext_lev10_ann_vol`, `ext_lev10_max_dd`, `ext_offset_sharpe_min`, `ext_offset_sharpe_max`, `ext_offset_mdd_worst`, `ext_sub1_sharpe_a`, `ext_sub2_sharpe_a`, `ext_fomc_n`, `ml_base_rate`. All 44 new keys appear in the appendix table automatically.

## Caveats to know before the rerun

- **FOMC duplicates fixed after the test run.** On some days the Fed issues several press releases (implementation note, January statement on longer-run goals). The test run sometimes used the short side document. The notebook now keeps one statement per day (the longest text = the main statement), so 192 statements become fewer, and the FinBERT rows can change slightly on the rerun. Given an AUC near 0.5, the conclusion will not change; check it anyway.
- **Sharpe ratio of a few hundredths ≠ a better strategy.** Even variant H's +0.05 is within the noise; the gates, not significance, decide.
- **Classifier sample starts in 2010** (burn-in), so the financial crisis is not part of its evaluation. Its stress gate therefore uses Covid and 2022 and needs both.
- **FinBERT measures tone about the economy, not hawkish/dovish policy stance.** The Fed-specific FOMC-RoBERTa model needs a Hugging Face login and was not used. FinBERT's licence is non-commercial (CC BY-NC-SA 4.0), which is fine for a course project.
- **Yahoo Finance revises adjusted prices**, so a rerun can differ in the last decimal (as the notebook already notes).
- **Not implemented:** market breadth from individual stocks (needs point-in-time S&P 500 constituents, e.g. CRSP via WRDS, to avoid survivorship bias).

## Checklist

- [ ] Rerun the notebook from a clean kernel; all asserts pass; Table 2 unchanged
- [ ] Check that the FOMC rows in `groupz_test_results.txt` still show no added value after the duplicate fix
- [ ] Add `transformers`, `torch`, `beautifulsoup4`, `requests` to `requirements.txt`
- [ ] Update prospectus §9 and §8 (text above), optionally §6; cite new numbers via the register keys
- [ ] Add the new references below to the bibliography
- [ ] Rebuild the prospectus PDF and the executed notebook PDF

## References (new)

- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. arXiv:1908.10063.
- Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
- López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. *Journal of Portfolio Management*, 42(4), 59–69.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Politis, D. N. and Romano, J. P. (1992). A circular block-resampling procedure for stationary data. In R. LePage and L. Billard (eds.), *Exploring the Limits of Bootstrap*, Wiley, 263–270.
