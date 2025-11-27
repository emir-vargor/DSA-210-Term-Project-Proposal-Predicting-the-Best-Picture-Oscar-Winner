#  DSA 210 Term Project: Predicting the Best Picture Oscar Winner

##  Motivation and Overview

The Academy Awards (the Oscars) are one of the most influential awards in the movie industry. A 'Best Picture' win provides significant marketing value and cultural prestige; therefore, the award is highly profitable and consequential for the moviemakers.

As a movie lover, I have always followed the award season. However, in recent years I've often felt that the winners follow a 'pattern' and that the award isn't just about film quality. This project was born from a desire to investigate this "pattern" using a data-driven approach.

In this project, a machine learning model will be trained on historical nominees to see if a win can be predicted using a set of quantifiable features. These features are:

* Genre
* Runtime
* Release date
* Critic score (MetaCritic)
* Performance in the Box Office
* Whether the movie is nominated for Best Director


This project has two objectives: first, to build a predictive model, and second, to analyze its features to determine if there are any "requirements" a film must fulfill to win 'Best Picture'. Since the nominees for the 2026 Oscars are expected to be announced after the deadline of this project, I plan to use the data between the years 1990-2020 to train my machine learning model, which will predict the winners of 2021-2025 accordingly. I expect to find that the chances for a nominated movie to win Best Picture vary according to its quantifiable features.

##   Null Hypothesis

**H₀**: The quantifiable features of a film have no statistically significant predictive power in determining the Best Picture winner.

## Data Collection and Methods

As stated in the project guidelines I will use a primary public dataset from “kaggle.com” and enrich it with several others. In this project, I will use Python (Pandas) to get the data. If needed, the websites will be used with the web scraping and data cleaning methods.

In order to collect the data for genre, runtime, release date, and whether the movie is nominated for Best Director, I will use:

**Datasets:**
* [https://www.kaggle.com/datasets/viniciusno/oscar-nominees-and-winners-1929-present](https://www.kaggle.com/datasets/viniciusno/oscar-nominees-and-winners-1929-present) is my based dataset, which shows Oscar nominees and winners. Additionally, it provides the data for total nominations.
* [https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows) is used to get the genre, runtime, and the MetaCritic score of the movie.
* These two datasets are merged with Python (Pandas). Since, some of the best picture nominated movies are not in iMDB top 1000 or some movie names do not match with each other in the datasets, the missing data is filled with web searching.

**Websites:**
* [https://www.imdb.com/](https://www.imdb.com/) is used t get missing genre, runtime and the MetaCritic data.


In order to collect the data for studio, and performance in the Box Office, I will use these dataset and website:
* [https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) gives the data for both box office revenue and the budget.
* [https://www.boxofficemojo.com/](https://www.boxofficemojo.com/)



## Data Analysis and Results
* **Genre**
Given that movies can be classified into multiple genres, Fisher's Exact Test was employed to analyze the dependency between genre and award outcomes. The resulting $p$-values for each genre ranged from $0.115$ to $1.0$. Consequently, we fail to reject the null hypothesis, indicating no statistically significant relationship between a movie's genre and its likelihood of winning the Best Picture award.

* **Runtime**
Initially, an independent samples t-test yielded a p-value of 0.1467 ($p > 0.05$), failing to reject the null hypothesis. Subsequently, using quantile binning produced a lower p-value of 0.066. While this result suggests a stronger association than the t-test, it remains statistically insignificant. Therefore, we conclude that there is no significant relationship between a movie's runtime and its likelihood of winning an award.

* **MetaScore**
A paired t-test yielded a p-value of 0.0099 ($p < 0.05$), leading to the rejection of the null hypothesis. This indicates a statistically significant difference in critical reception, with Oscar winners consistently achieving higher Meta Scores than their yearly rivals. However, having the highest score is not a guarantee; the single highest-rated movie of the year wins the award only 33.3% of the time.

* **Release Date**
The Chi-Square test yielded a p-value of 0.4551 ($p > 0.05$), failing to reject the null hypothesis. Consequently, we find no statistically significant association between a movie's release quarter and its likelihood of winning Best Picture.

* ** Best Director Nomination**
Chi-Square test yielded a p-value of 0.00079 ($p < 0.05$), providing strong evidence to reject the null hypothesis. This indicates a statistically significant dependency between the two variables; a movie's probability of winning Best Picture is heavily dependent on whether it also secured a nomination for Best Director.

## Possible Limitations

* **Political Context:** Political issues of the time the award was given may affect the winner drastically. The award may be given to a movie that reflects the political view of the academy, if so it could be very hard to detect.
* **Filmmaker Reputation:** Reputation of the filmmaker is another possible contributor factor for a movie to win the award. A renowned director with many Oscar nominations but could not win an award might be a step ahead for the award.
