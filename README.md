#  DSA 210 Term Project: Predicting the Best Picture Oscar Winner

##  Motivation and Overview

The Academy Awards (the Oscars) are one of the most influential awards in the movie industry. A 'Best Picture' win provides significant marketing value and cultural prestige; therefore, the award is highly profitable and consequential for the moviemakers.

As a movie lover, I have always followed the award season. However, in recent years I've often felt that the winners follow a 'pattern' and that the award isn't just about film quality. This project was born from a desire to investigate this "pattern" using a data-driven approach.

In this project, a machine learning model will be trained on historical nominees to see if a win can be predicted using a set of quantifiable features. These features are:

* Genre
* Runtime
* Release date
* Critic score (MetaCritic and Rotten Tomatoes)
* Public rating (IMDb and Letterboxd)
* Studio (Major/Indie/Streaming Platform)
* Performance in the Box Office
* Awards collected in the award season and festivals (Golden Globes, SAG, Cannes)
* Age rating
* Number of Oscar nominations in all categories
* Whether the movie is nominated for:
    * Best Director
    * Best Original/Adapted Screenplay
    * Best Actor/Actress

This project has two objectives: first, to build a predictive model, and second, to analyze its features to determine if there are any "requirements" a film must fulfill to win 'Best Picture'.

Since the nominees for the 2026 Oscars are expected to be announced after the deadline of this project, I plan to use the data between the years 1990-2020 to train my machine learning model, which will predict the winners of 2021-2025 accordingly. I expect to find that the chances for a nominated movie to win Best Picture vary according to its quantifiable features.

##  hypotheses Null Hypothesis

 **H₀**: The quantifiable features of a film have no statistically significant predictive power in determining the Best Picture winner.

##  Data Collection and Methods

As stated in the project guidelines I will use a primary public dataset from “kaggle.com” and enrich it with several others. In this project, I will use Python (Pandas) to get the data. If needed, the websites will be used with the web scraping and data cleaning methods.

In order to collect the data for genre, runtime, release date, public rating, age rating, awards collected in the award season and festivals, number of Oscar nominations, and whether the movie is nominated for: Best Director, Best Original/Adapted Screenplay, Best Actor/Actress, I will use:

**Datasets:**
* [https://www.kaggle.com/datasets/viniciusno/oscar-nominees-and-winners-1929-present](https://www.kaggle.com/datasets/viniciusno/oscar-nominees-and-winners-1929-present) is my based dataset, which shows Oscar nominees and winners. Additionally, it provides the data for total nominations.
* [https://www.kaggle.com/datasets/unanimad/golden-globe-awards](https://www.kaggle.com/datasets/unanimad/golden-globe-awards) gives me the opportunity to enrich my dataset.
* [https://www.kaggle.com/code/mollygarman/recent-oscar-winners](https://www.kaggle.com/code/mollygarman/recent-oscar-winners) gives clear data for recent Oscar winners, which is highly suitable for my project since it narrows down the set of movies.

**Websites:**
* [https://www.imdb.com/](https://www.imdb.com/)
* [https://letterboxd.com/](https://letterboxd.com/)
* [https://www.rottentomatoes.com](https://www.rottentomatoes.com)

In order to collect the data for studio, and performance in the Box Office, I will use these dataset and website:
*
