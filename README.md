# RateMyProfessor Analytics

Started as a simple tool: type in a professor's name and school, and get back their ratings, difficulty, courses taught, grade distributions, and an AI-generated summary of what students actually say about them. Then extended into a real dataset analysis: batch-scraped 57 NJIT professors across 12 departments to check whether course difficulty predicts ratings, whether review sentiment actually matches star ratings, and whether a model can predict a professor's rating from a handful of features.

Built with Python, the RateMyProfessor GraphQL API, VADER sentiment analysis, and scikit-learn.

## Part 1: Single-professor lookup tool

Search any professor at any school and get:
- Their overall quality and difficulty ratings, visualized as distributions
- Every course they've taught, ranked by number of reviews
- Letter grade distributions, both overall and broken down course-by-course
- "Would take again" percentage
- Descriptive statistics (mean, median, mode, standard deviation) on their ratings
- A 3-4 sentence AI-generated summary of their reviews (via Groq's API), covering teaching style, grading difficulty, and whether students recommend them

This part is fully generic - the school filter is just a text match against whatever you type in, so it works for any school RateMyProfessor covers, not just NJIT.

## Part 2: Batch dataset analysis

The single-professor tool is interactive and only looks at one person at a time, which isn't much of an analysis on its own. So I built a non-interactive version of the search function, looped it across 57 real NJIT professors spanning 12 departments (pulled from NJIT's official faculty directory), and used the resulting dataset to actually test some hypotheses.

**What it covers:**
- **Department comparison:** does average rating really differ by department, or is that just noise?
- **Difficulty vs. rating:** a Pearson correlation testing the "easy A" hypothesis directly
- **A within-department check:** does the difficulty effect still hold up inside a single department, or is it purely a between-department pattern?
- **Sentiment analysis:** running VADER on every scraped review's text, then checking whether the tone of what students write matches their numeric rating
- **A predictive model:** linear regression and random forest, predicting rating from difficulty, sentiment, and review count, benchmarked against a mean-only baseline

## What I found

Department differences in rating are real, but they're mostly a difficulty effect wearing a department's name. Humanities, English, and History average 4.1-4.5 out of 5, while Computer Science, Architecture, and Management average around 2.8. Difficulty follows almost the same ranking in reverse (Architecture and CS are the hardest-rated departments, English and Humanities the easiest), and difficulty correlates with rating at r = -0.74. So the honest read isn't "CS professors are worse teachers than English professors" - it's "CS courses are harder, and harder courses get rated lower, regardless of department."

That said, the effect doesn't clearly hold up within a single department. Inside Computer Science specifically (10 professors), difficulty and rating show basically no relationship (r = -0.045). With only 10 professors, that's probably a sample size problem rather than real evidence the effect vanishes - worth being upfront about rather than treating it as a clean null result.

What students write matches what they click. Average review sentiment correlates with numeric rating at r = 0.94 - about as strong a relationship as you'll see in real behavioral data. Reviews that read positively come with high star ratings, and vice versa. Worth noting VADER is a fairly simple tool, so part of this might just be that people who use strongly charged language are also the ones who leave extreme ratings, rather than this proving something deeper about how ratings form.

A model combining difficulty, sentiment, and review count explains about 90% of rating variance - but almost all of that comes from sentiment alone (feature importance of 0.93 out of 1.0). Difficulty and review count barely move the needle once sentiment is in the model. So this isn't really a new finding on top of the sentiment correlation - it's more confirmation that sentiment alone is already capturing nearly everything the numeric rating reflects.

## Limitations

57 professors across 12 departments is a small sample, and some departments only have 1-3 professors, which limits how much weight the department-level comparison can carry. The within-department check is almost certainly underpowered rather than a genuine null result. VADER is a lexicon-based sentiment tool, not a trained model, so the strong sentiment-rating correlation should be read with that in mind. And the predictive model was evaluated on a single train/test split given the small dataset size, so its R² and MAE are rough estimates, not a robust benchmark.

## Running it

Open the notebook in Google Colab (it's built around Colab-specific features like secrets management for the Groq API key). Run the setup cells at the top, then either:
- Use the interactive search cell to look up any single professor, or
- Skip to the batch analysis section, swap in your own list of (name, school) pairs, and re-run the EDA/sentiment/modeling cells on your own dataset

You'll need a free Groq API key (for the AI summary feature) stored as a Colab secret named `RMP`.

**Note on the AI summary feature:** it calls Groq's API with a specific model name (`openai/gpt-oss-120b` as of this writing). Groq periodically deprecates older models - if the AI summary cell starts throwing a `model_not_found` error, check [console.groq.com/docs/deprecations](https://console.groq.com/docs/deprecations) for a current model name and swap it into that cell.
