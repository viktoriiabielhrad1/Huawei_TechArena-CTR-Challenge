From the Oct 14 opening webinar (slides + Q&A)
-

The challenge, in short:
-

Build a model that predicts what a listener will do next, based only on their past interactions (user ID, item ID, timestamps).

Timeline:
- Phase 1: Oct 21 (9am Irish time) – Nov 4, midnight
- Top 15 teams announced: Nov 9
- Phase 2 (on-site, Dublin): Nov 14–15
- Awards ceremony: Nov 15

Phase 1 - Predict Clicks
-

- Given: a listener’s interaction history + a list of candidate items
- Task: predict the probability (0–1) that they’d click each candidate
- Submit: a CSV file of predicted probabilities
- No compute restrictions, any AI tools allowed
- Up to 10 submissions/day, live public leaderboard

Phase 2 — Compete and Adapt (on-site, top 15 teams only)
-

-  Our "agent" competes against other teams' agents, each offering a different item to the same simulated listener
- The simulated user picks one offer; we get the result back
- 4 rounds: develop > submit > evaluate > get new data > improve > repeat
- Submission is executable code/a trained model, not a CSV
- Huawei provides VMs with GPUs on-site - no need for personal compute, but bring a personal laptop to connect remotely
- Hard wall-clock time limit for evaluation (partial credit if incomplete)
- Stricter AI-tool rules apply; winning teams may be asked to explain their approach live

Prizes(split among members):
-

1st: €6,000 (1 team)
2nd: €2,500 (2 teams)
3rd: €1,500 (3 teams)

---------------------------------------------------------------------------------------------

Dataset/starter code released with Phase 1 on Oct 21

Replay of the webinar + slides will be posted on the platform

Update (25 Oct): Step2 - adding history length as a feature; time_bucket/day_type showed basically no standalone effect, so we keep them but shouldn’t over-invest there

---------------------------------------------------------------------------------------------

Starter solution - what needs to change - steps

-----------------------------------------------------------------------------------------------

Step 1. Add a time-based train/validation split - Viktoriia
-

Where: right after reading train.csv the first time (the “pass 1” loop that builds item_clicks, item_impr, etc.)

What: split by timestamp - fit (build counters) on Jan 20–25, hold out Jan 26–27 as a local test set. Score the held-out rows and compute ROC AUC against their real clicked values. Gives us a local score to check ideas against without burning a real submission.

Step 2. Build a per-row feature table for both train and eval - Dhurgham 
-

Where: replaces the score_row() function — specifically the combo_logit = ... line and the CF_WEIGHT * cf_score(...) line inside it.

What: instead of manually combining item_ctr, user_ctr, and cf_score into one number with a fixed formula, build a row of features per event/candidate — e.g. [item_ctr, user_ctr, cf_score, time_bucket, day_type, history_length or has_history], plus the real clicked label for training rows. Needs doing for every row of both train and eval — right now train.csv is only used to build counters, never turned into a labeled feature table itself.

Step 3. Make it run efficiently at scale - Sajjad 
-

Where: touches read_rows() and the feature-building loop from step 2.

What: train.csv is 7.3GB (6.8M rows), so building the feature table for every row needs proper chunked/streaming processing, efficient JSON parsing for the history columns, and ideally writing features out incrementally rather than holding everything in memory at once.

Step 4. Train XGBoost and generate final probabilities - Yasashwi 
-

Where: new code, goes after the feature table exists; replaces the line p = 1.0 / (1.0 + math.exp(-z)) in the pass-2 loop over evaluation.csv.

What: feed the labeled training feature table into XGBoost’s .fit(), so it learns how to weight the signals instead of a fixed CF_WEIGHT. Then call .predict_proba() on the evaluation feature table to get final probabilities directly — no manual sigmoid math needed.

Stays as-is: read_rows()’s CSV/JSON parsing logic, the counters (item_clicks, item_impr, etc.) and co-click table (co, item_pop_sqrt), and the logit()/smoothing formula — all reusable exactly as written, just feeding a model now instead of a formula.

Step 5. Keeping track of the project until the Nov 4 deadline, and presentation prep if we make top 15 - Jade
-
