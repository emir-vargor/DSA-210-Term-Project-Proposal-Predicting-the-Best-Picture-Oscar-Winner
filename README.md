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

Visuals can be found on "tests_and_graphs.ipynb". 


## Machine Learning Model
In this part, we use *the Random Forest algorithm*. The motivation behind this choice was the specific nature of our dataset and the prediction task:
1.   **Handling Small Datasets and Overfitting:** Our training dataset consists of approximately 193 samples (N=193). With such limited data, Random Forest algorithm is highly suitable for avoiding overfitting (high variance), since the Random Forest algorithm is trained over different subsets of data and gives the average of the predictions, resulting in reduced variance and improved generalization.
2.   **Addressing Class Imbalance:** The Oscar prediction task involves a highly imbalanced dataset, where the minority class ('Winner') is significantly smaller than the majority class ('Non-Winner'). Random Forest allows us to effectively handle this imbalance by utilizing the "class_weight='balanced'" parameter. This parameter leads the model to penalize misclassifying the minority class more heavily.

1.   **Feature Importance and Interpretability:** Random Forest provides interpretability through Feature Importance scores. This is crucial for our analysis, as it allows us to quantify the impact of specific features (such as PGA Awards or Genre) on a film's probability of winning.  

&nbsp;
&nbsp;
Then, I design two ML models. First one is the baseline model, that is created for the purpose of seeking the parts that needs to be improved. The results of this model are considerably poor, since the baseline model doesnot encompass various enriching, purifying and, refining methods. Then, I used these tecniques to strenghen the model and create an Advanced ML Model: Natural Language Processing (NLP) with TF-IDF, Feature Selection via RFE, Dimensionality Reduction with PCA, Model Optimization with GridSearchCV & Cross-Validation.  

### Advanced Machine Learning Model

In order to strenghening our dataset we used these tecniques:


1.   **Natural Language Processing (NLP) with TF-IDF:** In order to effectively utilize the categorical 'Genre' data, we applied TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, simple One-Hot Encoding. This technique assigns weights to genre tags based on their frequency across the dataset.In this way, the model can distinguish between common genres (e.g., 'Drama') and rarer, potentially more distinguishing genres (e.g., 'Musical'), enriching the feature space with semantic importance.
2.  **Feature Selection via RFE:** Given the limited size of our dataset (N=193), reducing dimensionality was crucial to prevent overfitting. We employed Recursive Feature Elimination (RFE) to iteratively discard the least significant features. This allowed us to identify and retain only the top 10 most predictive features, and resulted in reducing noise and improving the model's generalization capability.

1.   **Dimensionality Reduction with PCA:** We utilized Principal Component Analysis (PCA) as an unsupervised learning technique for Exploratory Data Analysis (EDA). By projecting the high-dimensional feature space into two principal components, we visualized the separability of the 'Winner' and 'Non-Winner' classes. This visualization provided insights into the intrinsic difficulty of the classification task and the clustering behavior of the data.
2.   **Model Optimization with GridSearchCV & Cross-Validation:** In order to ensure the robustness of our Random Forest model, we implemented Stratified 5-Fold Cross-Validation. This was particularly important due to the class imbalance, as it ensured that each fold contained a representative ratio of winners. Furthermore, we used GridSearchCV to systematically tune hyperparameters (e.g., max_depth, n_estimators), finding the optimal trade-off between bias and variance.  

&nbsp;
&nbsp;

Our dataset is small and imbalanced. To avoid overfitting (or memorization), we need to filter out noise using RFE and handle categorical data using TF-IDF. These techniques are important to turn a weak model into a robust one.  
Top three Predictive Features: "pga_winner"(%29), "Nominated_Both_Director_and_Picture"(%17), "Golden_Globe_Picture_Winner"(%13)


### Evaluation of the Advanced ML Model
***Accuracy (%92)**: A very high rate of 92% has been achieved. This indicates that the model correctly classifies the vast majority of films. *(8% increase over the baseline model)*

**Recall (%80)**: Our model correctly identified 4 out of the 5 actual winners in the test set, proving how capable the model is at "not missing the winner. *(300% increase over the baseline model)*


**Precision (0.57)**: Since our model does not flag just one, but top two or three movies with the highest potential to win as "likely winners." These "False Positives" are not errors, instead they are strong contenders. *(72% increase over the baseline model)*

**Confusion Matrix Analysis**: The matrix reveals that our model successfully identified 4 out of 5 actual winners (True Positives), achieving a high Recall of 0.80. The single missed winner (False Negative = 1) corresponds to CODA, which we have identified as a statistical outlier (in the Generalization Capability part). Furthermore, the low number of False Positives (3) indicates that when the model predicts a win, it is highly likely to be a strong contender. Overall, the matrix confirms the model's reliability in distinguishing Oscar-worthy films.
  
**F1-Score: (0.67)**: Since our dataset is highly imbalanced, F1-Score of our model is not significantly low. In fact, this score demonstrates that our model has reached a balanced structure in terms of both making accurate predictions and avoiding unnecessary false alarms. *(168% increase over the baseline model)*

 
**Interpretation of AUC Score:** Since our AUC score (0.93) is close to 1.0 indicates that the model has a high degree of separability, meaning it assigns higher probabilities to actual winners than to non-winners in almost all cases. Hence, our high AUC score confirms that the model ranks the candidates correctly with accuracy.  

**The Probability Distribution Plot Analysis**:The Probability Distribution Plot reveals a sharp separation between classes. Non-winners are densely clustered near 0.0, showing effective filtering, while winners are shifted towards the high-probability region (0.6–1.0). This distinct split confirms our model is not merely guessing but confidently identifies true contenders based on robust learned signals.

&nbsp;
&nbsp;
### Generalization Capability  
**2024 (Oppenheimer) & 2021 (Nomadland)**: The model placed the undisputed favorites of these years at rank #1 with very clear probability margins, such as 87% and 61%. This proves that the model has learned strong signals (Award history, Director, Critic scores) very well.  

**2023 (Everything Everywhere All at Once)**: Thanks to the TF-IDF technique, our model can analyze films with cross-genre characteristics. Hence the winning probability increased to 53%,  which was 29% according to the baseline model.  

**2022 (CODA)**: CODA is an outlier in our dataset since it is only the sixth film that won the Best Picture award without the Best Director nomination. Additionally, the movie did not win a Golden Globe. Hence our model putting CODA at 4th place indicates its robustness, since it did not get carried away by this "Noise" and remained loyal to the statistics.   

**2025 (Anora)**: As seen in the table, the movie shares the strong statistical characteristics of past winners like Oppenheimer and Nomadland (Director nomination and key precursor awards). Consequently, the model correctly identified it as the clear favorite.

&nbsp;
&nbsp;
In conclusion; enriched with TF-IDF, purified of noise via RFE, and refined through imbalance management, our model has successfully modeled a chaotic and human-centric problem like the Oscars with 92% accuracy and high consistency.  
More explanation and visualization related to ML models can be found on "Machine_Learning.ipynb".

## Conclusion
* **Rejecting the Null Hypothesis:** Based on both statistical testing and the success of the Machine Learning model, we reject the Null Hypothesis ($H_0$). Quantifiable features do have significant predictive power.
* The initial suspicion that winners follow a pattern is true, but it is not necessarily a pattern of content (Genre, Runtime). Instead, it is a pattern of Industry Consensus. Critical Quality and performance in the award season are the deciding factors. In conclusion, winning Best Picture is less about the type of movie you make and more about securing the specific approval of industry guilds (Directors and Producers) and maintaining high critical acclaim.


## AI Usage
This project utilized Generative AI tools (specifically Google Gemini) to assist in the documentation, coding and debugging phases. Specific prompts are shared in ".ipynb" files.

## Limitations
* **Oscar Bump:** I initially intended to analyze the correlation between financial success and award chances, as many industry experts attribute the underperformance of films like Babylon (2022) at the Oscars to their poor box office results. However, as explained in the project proposal, a 'Best Picture' win provides significant marketing value and reputation. Therefore, the model excludes box office and IMDb rating data to avoid data leakage, as ticket sales and public rating often increase significantly after nominations and award wins. Using total gross would imply that the model has access to future information that was not available at the time of prediction. 
* **Filmmaker Reputation:** Reputation of the filmmaker is another possible contributor factor for a movie to win the award. A renowned director with many Oscar nominations but could not win an award might be a step ahead for the award.
* **Politics and Campaigning:** The voting behavior of the Academy is often influenced by unquantifiable external factors, such as aggressive marketing campaigns, studio politics, and the prevailing socio-political climate during the voting period. Like in the example of Shakespeare in Love (1998) winning over the heavy favorite Saving Private Ryan, a result widely attributed to Miramax's unprecedented and aggressive campaigning tactics.

## Future Work
* **Adding More Award Shows:** Including data from BAFTA and Critics Choice Awards would help the model understand international trends better, not just the American ones.
* **Trying Different Models:** I used the Random Forest model for this project. Later, I could test other advanced models like XGBoost or LightGBM to see if they can predict the winners more accurately.
