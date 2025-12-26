#  DSA 210 Term Project: Predicting the Best Picture Oscar Winner

##  Motivation and Overview

The Academy Awards (the Oscars) are one of the most influential awards in the movie industry. A 'Best Picture' win provides significant marketing value and cultural prestige; therefore, the award is highly profitable and consequential for the moviemakers.

As a movie lover, I have always followed the award season. However, in recent years I've often felt that the winners follow a 'pattern' and that the award isn't just about film quality. This project was born from a desire to investigate this "pattern" using a data-driven approach.

In this project, a machine learning model will be trained on historical nominees to see if a win can be predicted using a set of quantifiable features. These features are:

* Genre
* Runtime
* Release date
* Critic score (MetaCritic)
* Whether the movie is nominated for Best Director
* Whether the movie won Best Motion Picture - Drama  or Best Motion Picture - Musical or Comedy in Golden Globes
* Whether the movie won Best Theatrical Motion Picture award in Producers Guild of America Awards


This project has two objectives: first, to build a predictive model, and second, to analyze its features to determine if there are any "requirements" a film must fulfill to win 'Best Picture'. Since the nominees for the 2026 Oscars are expected to be announced after the deadline of this project, I plan to use the data between the years 1990-2020 to train my machine learning model, which will predict the winners of 2021-2025 accordingly. I expect to find that the chances for a nominated movie to win Best Picture vary according to its quantifiable features.

##   Null Hypothesis

**H₀**: The quantifiable features of a film have no statistically significant predictive power in determining the Best Picture winner.

## Data Collection and Methods

As stated in the project guidelines I will use a primary public dataset from “kaggle.com” and enrich it with several others. In this project, I will use Python (Pandas) to get the data. If needed, the websites will be used with the web scraping and data cleaning methods.

In order to collect the data for genre, runtime, release date, and whether the movie is nominated for Best Director, I will use:

**Datasets:**
* https://www.kaggle.com/datasets/viniciusno/oscar-nominees-and-winners-1929-present is my based dataset, which shows Oscar nominees and winners.
* https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows is used to get the genre, runtime, and the MetaCritic score of the movie.
* https://www.kaggle.com/code/isaienkov/full-eda-and-visualization-golden-globe-awards/input is used to get the data for the Golden Globes Awards. For the ceremonies after 2020, Golden Globes data is added manually.
* These three datasets are merged with Python (Pandas). Since, some of the best picture nominated movies are not in iMDB top 1000 or some movie names do not match with each other in the datasets, the missing data is filled with web searching.

**Websites:**
* [https://www.imdb.com/](https://www.imdb.com/) is used to get missing genre, runtime and the MetaCritic data.
* https://en.wikipedia.org/wiki/Producers_Guild_of_America_Awards is used to get PGA winning data. A . csv file that encompasses only the winners is crafted by hand, then merged with the main dataset. 




## Data Analysis and Results
* **Genre**
Given that movies can be classified into multiple genres, Fisher's Exact Test was employed to analyze the dependency between genre and award outcomes. The resulting $p$-values for each genre ranged from $0.11$ to $1.0$. Consequently, we fail to reject the null hypothesis, indicating no statistically significant relationship between a movie's genre and its likelihood of winning the Best Picture award.

* **Runtime**
Initially, an independent samples t-test yielded a p-value of 0.1467 ($p > 0.05$), failing to reject the null hypothesis. Subsequently, using quantile binning produced lower p-values. The lowest being $p = 0.0235$ > $0.0038$ (A corrected p-value according to the Bonferroni correction.) . While this result suggests a stronger association than the t-test, it remains statistically insignificant. Therefore, we conclude that there is no significant relationship between a movie's runtime and its likelihood of winning an award.

* **MetaScore**
A paired t-test yielded a p-value of 0.0433 ($p < 0.05$), leading to the rejection of the null hypothesis. This indicates a statistically significant difference in critical reception, with Oscar winners consistently achieving higher Meta Scores than their yearly rivals. Moreover, subsequent binned analysis with Bonferroni correction revealed that no specific score range guarantees a win, reinforcing the idea that high scores are necessary but insufficient on their own.

* **Release Date**
The Chi-Square test yielded a p-value of 0.4551 ($p > 0.05$), failing to reject the null hypothesis. Consequently, we find no statistically significant association between a movie's release quarter and its likelihood of winning Best Picture.

* **Best Director Nomination**
Chi-Square test yielded a p-value of 0.000265 ($p < 0.05$), providing strong evidence to reject the null hypothesis. This indicates a statistically significant dependency between the two variables; a movie's probability of winning Best Picture is heavily dependent on whether it also secured a nomination for Best Director. As stated in the Machine Learning part, in 2022 the movie CODA stands as an oulier, since the movie won the Best Picture without dual nominaion.

* **Golden Globe Winning** 
We hypothesized a significant correlation between winning a Golden Globe and the Best Picture Oscar, testing this by comparing winners of both awards using Fisher's Exact Test due to our small sample size (4). The test yielded a p-value of 0.000045, far below the 0.05 threshold, which leads us to reject the null hypothesis; this extremely low value confirms that the overlap in winners is not random, but rather that winning a Golden Globe is a statistically significant predictor of Oscar success.

* **Producers Guild of America Awards Winning**
A Fisher's Exact Test on the relationship between winning a PGA award and winning Best Picture yielded a p-value of 0.00000... (almost zero) ($p < 0.05$) and an Odds Ratio of 41.95. This extremely strong statistical significance leads us to reject the null hypothesis. The analysis confirms that the PGA Award is the single most critical indicator of Oscar success in this study.

## Machine Learning Model
To operationalize these findings, I built a machine learning model to predict the Best Picture winner.
* Model Selection: A Random Forest Classifier was chosen for its ability to handle tabular data, manage interactions between features (e.g., the combination of a high MetaScore and a PGA win), and provide feature importance rankings.
* Training & Testing Split: Training Set is set of movies from 1990 to 2020. Testing Set is set of movies from 2021 to 2025.
* Features Used: Runtime, MetaScore (MinMax Scaled). Boolean Flags: Nominated_Both_Director_and_Picture, Golden_Globe_Picture_Winner, pga_winner. Genre (One-Hot Encoded).
* Model Results (2021–2025 Performance) : Predict correctly for 2021,2023,2024,2025. For 2022, CODA stood out as an anomaly, becoming only the sixth film in history to win Best Picture without a Best Director nomination.
* Feature Importance: PGA Winner (32%), MetaScore (14%), Runtime (13%), Golden Globe Winner (9%)


## Conclusion
* **Rejecting the Null Hypothesis:** Based on both statistical testing and the success of the Machine Learning model, we reject the Null Hypothesis ($H_0$). Quantifiable features do have significant predictive power.
* The initial suspicion that winners follow a pattern is true, but it is not necessarily a pattern of content (Genre, Runtime). Instead, it is a pattern of Industry Consensus. Critical Quality and performance in the award season are the deciding factors. In conclusion, winning Best Picture is less about the type of movie you make and more about securing the specific approval of industry guilds (Directors and Producers) and maintaining high critical acclaim.


## AI Usage
This project utilized Generative AI tools (specifically Google Gemini) to assist in the documentation, coding and debugging phases. Specific prompts are shared in .ipynb files.

## Limitations
* **Oscar Bump:** I initially intended to analyze the correlation between financial success and award chances, as many industry experts attribute the underperformance of films like Babylon (2022) at the Oscars to their poor box office results. However, as explained in the project proposal, a 'Best Picture' win provides significant marketing value and reputation. Therefore, the model excludes box office and IMDb rating data to avoid data leakage, as ticket sales and public rating often increase significantly after nominations and award wins. Using total gross would imply that the model has access to future information that was not available at the time of prediction. 
* **Filmmaker Reputation:** Reputation of the filmmaker is another possible contributor factor for a movie to win the award. A renowned director with many Oscar nominations but could not win an award might be a step ahead for the award.
* **Politics and Campaigning:** The voting behavior of the Academy is often influenced by unquantifiable external factors, such as aggressive marketing campaigns, studio politics, and the prevailing socio-political climate during the voting period. Like in the example of Shakespeare in Love (1998) winning over the heavy favorite Saving Private Ryan, a result widely attributed to Miramax's unprecedented and aggressive campaigning tactics.

## Future Work
* **Adding More Award Shows:** Including data from BAFTA and Critics Choice Awards would help the model understand international trends better, not just the American ones.
* **Trying Different Models:** I used the Random Forest model for this project. Later, I could test other advanced models like XGBoost or LightGBM to see if they can predict the winners more accurately.
