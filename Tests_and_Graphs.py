import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Load and Filter
df = pd.read_csv('Final_best_picture_data.csv')
YEAR_COL = "year_ceremony"
df_filtered = df[df[YEAR_COL] <= 2020].copy() # since the ceremonies after 2020 are to be used to test our model

df_filtered['release_date'] = pd.to_datetime(df_filtered['release_date'])
df_filtered['quarter'] = df_filtered['release_date'].dt.quarter
df_filtered['runtime_int'] = df_filtered['Runtime'].astype(str).str.replace(' min', '').astype(int)

#### QUARTER
# 2. Analyzing Quarters
quarter_stats = df_filtered.groupby(['quarter', 'winner']).size().unstack(fill_value=0)
quarter_stats.columns = ['Nominees', 'Winners']
print("--- Quarter Stats ---")
print(quarter_stats)

# CHI-SQUARE Test
contingency_table = pd.crosstab(df_filtered['quarter'], df_filtered['winner']).reindex([1, 2, 3, 4], fill_value=0)
contingency_table.index = ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"]
contingency_table.columns = ["Nominee (Lost)", "Won"]

print("--- OBSERVED VALUES ---")
print(contingency_table)
contingency_table_test = contingency_table[contingency_table.sum(axis=1) > 0]
print("\n" + "="*40 + "\n")

chi2, p, dof, expected = stats.chi2_contingency(contingency_table_test)

# Results
print(f"Chi-Square Statistics: {chi2:.4f}")
print(f"P-value:     {p:.4f}")
print(f"Degrees of Freedom:    {dof}")

# Interpretation
alpha = 0.05
print("\n--- ANALYSIS ---")
if p < alpha:
    print("Result: H0 is rejected. ")
    print("Interpretation: There is a significant relationship between Oscar winning and relase date of the movie.")
else:
    print("Result: We fail to reject H0.")
    print("Interpretation: There is not a significant relationship between Oscar winning and relase date of the movie.")

# Visualization

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

plot_data = df_filtered.groupby('quarter')['winner'].value_counts(normalize=False).rename('count').reset_index()
plot_data['Data'] = plot_data['winner'].map({True: 'Winner', False: 'Just a Nominee'})

# Plotting
sns.barplot(
    data=plot_data,
    x='quarter',
    y='count',
    hue='Data',
    palette={'Winner': 'gold', 'Just a Nominee': 'grey'}
)

plt.title('Number of Oscar Winnings in the Release Date Corresponding Quarter of the Year (1991-2020)')
plt.xlabel('Quarters of a Year')
plt.ylabel('Number of Movies')
plt.legend(title='Results')
plt.show()


#### GENRE
df_genres = df_filtered.assign(Genre=df_filtered['Genre'].str.split(', ')).explode('Genre')

genre_stats = df_genres.groupby(['Genre', 'winner']).size().unstack(fill_value=0)

genre_stats.columns = ['Losses', 'Wins']


genre_stats['Total_Nominees'] = genre_stats['Wins'] + genre_stats['Losses']


total_winners_global = genre_stats['Wins'].sum()
total_losses_global = genre_stats['Losses'].sum()
total_nominees_global = total_winners_global + total_losses_global

print(f"Total Nominees: {total_nominees_global}")
print(f"Total Winners: {total_winners_global}")
print("-" * 60)

# ---  FISHER'S EXACT TEST  ---

results = []

print(f"{'GENRE':<15} {'ODDS RATIO':<12} {'P-VALUE':<10} {'SIGNIFICANCE'}")
print("-" * 60)

# Iterate through each genre in the calculated statistics
for genre in genre_stats.index:
    
    # Get specific counts for the current genre
    genre_win = genre_stats.loc[genre, 'Wins']
    genre_loss = genre_stats.loc[genre, 'Losses']
    
    # Calculate the "Rest" (All other genres combined)
    # We subtract current genre stats from the global totals
    rest_win = total_winners_global - genre_win
    rest_loss = total_losses_global - genre_loss

    # Construct the 2x2 Contingency Table
    # [[Genre_Win, Genre_Loss],
    #  [Rest_Win,  Rest_Loss ]]
    table = [[genre_win, genre_loss], [rest_win, rest_loss]]

    # Apply Fisher's Exact Test
    odds_ratio, p_value = stats.fisher_exact(table, alternative='two-sided')

    significance = ""
    if p_value < 0.05:
        significance = "Significant"
    else:
        significance = "-"

    print(f"{genre:<15} {odds_ratio:.2f}         {p_value:.3f}      {significance}")

    results.append({
        'Genre': genre,
        'Odds_Ratio': odds_ratio,
        'P_Value': p_value,
        'Significance': significance
    })


#### RUNTIME ANALYSIS 

df_filtered['runtime_bin'] = pd.qcut(df_filtered['runtime_int'], q=5, labels=["Very Short", "Short", "Medium", "Long", "Very Long"])

runtime_stats = df_filtered.groupby(['runtime_bin', 'winner'], observed=False).size().unstack(fill_value=0)
runtime_stats_for_all = df_filtered.groupby('winner')['runtime_int'].agg(['mean', 'count', 'std'])
print("\n--- Runtime Stats ---")
print(runtime_stats_for_all)
print("\n")
print(runtime_stats)

# INDEPENDENT T-TEST FOR RUNTIME

runtimes_winners = df_filtered[df_filtered['winner'] == True]['runtime_int']
runtimes_nominees = df_filtered[df_filtered['winner'] == False]['runtime_int']

t_stat_run, p_val_run = stats.ttest_ind(runtimes_winners, runtimes_nominees, equal_var=False)

print(f"\n--- Runtime T-Test Results ---")
print(f"T-statistic: {t_stat_run:.4f}")
print(f"P-value:     {p_val_run:.4f}")

alpha = 0.05
if p_val_run < alpha:
    print("Result: Reject H0.")
    print("Interpretation: There is a significant difference in runtime between Winners and Nominees.")
else:
    print("Result: Fail to reject H0.")
    print("Interpretation: No significant difference in runtime found.")


# CHI-SQUARE TEST
runtime_contingency = pd.crosstab(df_filtered['runtime_bin'], df_filtered['winner'])

runtime_contingency.columns = ["Nominee (Lost)", "Winner"]

print("\n--- Observed Values (Quantile Bins) ---")
print(runtime_contingency)

chi2_run, p_run, dof_run, expected_run = stats.chi2_contingency(runtime_contingency)

print(f"\n--- Chi-Square Test for Runtime Quantiles ---")
print(f"Chi-Square Statistic: {chi2_run:.4f}")
print(f"P-value:              {p_run:.4f}")

# Interpretation
alpha = 0.05
print("\n--- ANALYSIS ---")
if p_run < alpha:
    print("Result: H0 is rejected.")
    print("Interpretation: There is a significant relationship between runtime intervals (quantiles) and winning an Oscar.")

    runtime_contingency['Win_Rate'] = runtime_contingency['Winner'] / (runtime_contingency['Winner'] + runtime_contingency['Nominee (Lost)'])

    best_bin = runtime_contingency['Win_Rate'].idxmax()
    highest_rate = runtime_contingency['Win_Rate'].max()

    print(f"Insight: The runtime interval with the highest win rate is {best_bin} (Win Rate: {highest_rate:.2%})")
    print("\nWin Rates per Quantile:")
    print(runtime_contingency['Win_Rate'].sort_values(ascending=False))
else:
    print("Result: We fail to reject H0.")
    print("Interpretation: There is no significant relationship between runtime intervals and winning an Oscar.")

# VISUALIZATION
plt.figure(figsize=(8, 6))
# Boxplot
sns.boxplot(
    data=df_filtered, 
    x='winner', 
    y='runtime_int', 
    hue='winner',          
    palette=['#b0b0b0', '#d62728'], 
    legend=False                     
)

plt.title('Distribution of Movie Runtimes: Winners vs Nominees', fontsize=14)
plt.xlabel('Is Winner?', fontsize=12)
plt.ylabel('Runtime (Minutes)', fontsize=12)

plt.xticks(ticks=[0, 1], labels=['Nominee (Lost)', 'Winner'])

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#### MetaScore

meta_stats = df_filtered.groupby('winner')['Meta_score'].agg(['mean', 'count', 'std'])
print("\n--- General Meta Score Stats (Global) ---")
print(meta_stats)


# --- YEARLY PAIRED ANALYSIS ---

yearly_meta = df_filtered.groupby(['year_ceremony', 'winner'])['Meta_score'].mean().unstack()
yearly_meta.columns = ['Avg_Score_Nominees', 'Winner_Score']

# Visualization
plt.figure(figsize=(14, 6))
sns.lineplot(data=yearly_meta, markers=True, dashes=False)
plt.title('Meta Score Trends: Winners vs. Average Nominees (1991-2020)')
plt.ylabel('Meta Score')
plt.xlabel('Ceremony Year')
plt.legend(title='Group')
plt.grid(True, alpha=0.3)
plt.show()


# --- PAIRED T-TEST ---

paired_clean = yearly_meta.dropna()
t_stat_meta, p_val_meta = stats.ttest_rel(paired_clean['Winner_Score'], paired_clean['Avg_Score_Nominees'])

print(f"\n--- Paired T-Test Results (Meta Score) ---")
print(f"T-statistic: {t_stat_meta:.4f}")
print(f"P-value:     {p_val_meta:.4f}")

alpha = 0.05
print("\n--- ANALYSIS ---")
if p_val_meta < alpha:
    print("Result: H0 is rejected (Significant).")
    print("Interpretation: There is a statistically significant difference.")
    if t_stat_meta > 0:
        print("Insight: Oscar Winners consistently have higher Meta Scores than their rivals.")
    else:
        print("Insight: Oscar Winners actually have LOWER scores (Rare/Unexpected).")
else:
    print("Result: Fail to reject H0.")
    print("Interpretation: Critics' scores do not significantly differentiate winners from nominees.")


# --- VISUALIZING THE "CRITIC GAP" ---
paired_clean['Score_Diff'] = paired_clean['Winner_Score'] - paired_clean['Avg_Score_Nominees']

plt.figure(figsize=(10, 6))
sns.histplot(paired_clean['Score_Diff'], kde=True, color='green', bins=15)
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='No Difference (0)')

plt.title('Distribution of Score Differences (Winner - Avg Nominee)')
plt.xlabel('Score Difference (Points)')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# --- "CRITICAL FAVORITE" ANALYSIS ---
print("\n--- Does the 'Critical Favorite' Win? ---")

max_scores_per_year = df_filtered.groupby('year_ceremony')['Meta_score'].transform('max')

winners = df_filtered[df_filtered['winner'] == True].copy()
winners['is_top_rated'] = winners['Meta_score'] == max_scores_per_year[winners.index]

top_rated_win_rate = winners['is_top_rated'].mean()
print(f"Percentage of time the highest-rated movie wins: {top_rated_win_rate:.2%}")

#### DUAL NOMINATION
print("\n" + "="*60)
print("STATISTICAL ANALYSIS: Director Nomination vs Best Picture Win")
print("="*60)

dir_nom_col = 'nominated_director_and_picture'

if dir_nom_col not in df_filtered.columns:
    print(f"\n[Error] Column '{dir_nom_col}' not found in the dataframe.")
    print("Please ensure you loaded the updated CSV file containing the director nomination flag.")
else:
    df_filtered[dir_nom_col] = df_filtered[dir_nom_col].astype(bool)

    contingency_dir = pd.crosstab(df_filtered[dir_nom_col], df_filtered['winner'])
    contingency_dir.index = ["No Director Nom", "Has Director Nom"]
    contingency_dir.columns = ["Lost BP", "Won BP"]

    print("\n--- Observed Values ---")
    print(contingency_dir)

    # Chi-Square Test
    chi2_d, p_d, dof_d, exp_d = stats.chi2_contingency(contingency_dir)
    print(f"\n--- Chi-Square Results ---")
    print(f"Chi-Square Statistic: {chi2_d:.4f}")
    print(f"P-value:              {p_d:.4g}")
    # Interpretation
    alpha = 0.05
    if p_d < alpha:
        print("\nResult: SIGNIFICANT (p < 0.05)")
        print("Interpretation: There is a statistically significant relationship.")
        print("Movies nominated for Best Director are significantly more likely to win Best Picture.")
        
        if "Has Director Nom" in contingency_dir.index:
            wins = contingency_dir.loc["Has Director Nom", "Won BP"]
            total = contingency_dir.loc["Has Director Nom"].sum()
            win_rate_with_nom = wins / total if total > 0 else 0
            print(f"\nWin Rate WITH Director Nom:    {win_rate_with_nom:.2%}")
        
        if "No Director Nom" in contingency_dir.index:
            wins = contingency_dir.loc["No Director Nom", "Won BP"]
            total = contingency_dir.loc["No Director Nom"].sum()
            win_rate_no_nom = wins / total if total > 0 else 0
            print(f"Win Rate WITHOUT Director Nom: {win_rate_no_nom:.2%}")
    else:
        print("\nResult: Fail to reject H0.")
        print("Interpretation: No significant relationship found.")
    # Visualization
    plt.figure(figsize=(10, 6))
    
    plot_df = df_filtered.copy()
    plot_df['Director_Nom_Status'] = plot_df[dir_nom_col].map({True: 'Nominated for Director', False: 'Not Nominated'})
    plot_df['Outcome'] = plot_df['winner'].map({True: 'Won Best Picture', False: 'Lost'})
    
    ax = sns.countplot(
        data=plot_df, 
        x='Director_Nom_Status', 
        hue='Outcome', 
        palette={'Won Best Picture': 'gold', 'Lost': 'grey'},
        hue_order=['Lost', 'Won Best Picture']
    )
    
    plt.title('Impact of Best Director Nomination on Best Picture Chances')
    plt.xlabel('Director Nomination Status')
    plt.ylabel('Number of Movies')
    plt.legend(title='Outcome')
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
             ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, color='black')
    
    plt.show()

#### GOLDEN GLOBES

warnings.filterwarnings("ignore")

def analyze_awards_correlation():
    # Load the dataset
    df = pd.read_csv('Final_best_picture_data.csv')

    # Data Cleaning 
    df['winner'] = df['winner'].astype(str).replace({'True': True, 'False': False, 'nan': False})
    df['golden_globe_winner'] = df['golden_globe_winner'].astype(str).replace({'True': True, 'False': False, 'nan': False})
    
    df['winner'] = df['winner'].map({True: True, False: False, 'True': True, 'False': False})
    df['golden_globe_winner'] = df['golden_globe_winner'].map({True: True, False: False, 'True': True, 'False': False})

    # Drop rows where either value might still be NaN (just in case)
    df = df.dropna(subset=['winner', 'golden_globe_winner'])

    # Create a Contingency Table (Crosstab)
    contingency_table = pd.crosstab(
        df['golden_globe_winner'], 
        df['winner'], 
        rownames=['Golden Globe Winner'], 
        colnames=['Oscar Winner (Best Picture)']
    )

    print("\n--- Contingency Table ---")
    print(contingency_table)

    # 4. Chi-Square Test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    test_name = "Chi-Square Test"

    print(f"\n--- Statistical Test Results ({test_name}) ---")
    print(f"P-value: {p_value:.5f}")
    if p_value < 0.05:
        print("Result: Statistically Significant Association (p < 0.05)")
        print("There is a significant relationship between winning a Golden Globe and winning the Best Picture Oscar.")
    else:
        print("Result: No Statistically Significant Association (p >= 0.05)")
        print("There is no strong evidence to suggest that winning a Golden Globe predicts the Oscar win.")

    # Visualizations
    plt.figure(figsize=(14, 6))

    # Plot 1: Heatmap of the Contingency Table
    plt.subplot(1, 2, 1)
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.title('Correlation Heatmap: Golden Globe vs. Oscar')
    
    # Plot 2: Win Rate Comparison
    # Calculate win rate given GG win vs No GG win
    win_rates = df.groupby('golden_globe_winner')['winner'].mean() * 100
    
    plt.subplot(1, 2, 2)
    bars = win_rates.plot(kind='bar', color=['#95a5a6', '#f1c40f'], alpha=0.8)
    plt.title('Oscar Win Probability based on Golden Globe Result')
    plt.ylabel('Probability of Winning Oscar (%)')
    plt.xlabel('Did the film win a Golden Globe?')
    plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
    plt.ylim(0, 100)
    
    # Add percentage labels
    for i, v in enumerate(win_rates):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_awards_correlation()