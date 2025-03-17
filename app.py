import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Charger les données (supposons que le contenu CSV est enregistré dans un fichier 'data.csv')
# Vous pouvez remplacer cela par votre méthode de chargement des données
save_path = sys.argv[1] if len(sys.argv) > 1 else '.'

df = pd.read_csv('./archive/deepseek_vs_chatgpt.csv')

# 1. Analyse descriptive par langue
print("Analyse descriptive de la précision par langue:")
language_stats = df.groupby('Language')['Response_Accuracy'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).sort_values(by='mean', ascending=False)
print(language_stats)

# 2. Visualisations
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Graphique à barres de la précision moyenne par langue
plt.subplot(1, 2, 1)
sns.barplot(x=language_stats.index, y=language_stats['mean'], palette='viridis')
plt.title('Précision moyenne par langue')
plt.xlabel('Langue')
plt.ylabel('Précision moyenne')
plt.xticks(rotation=45)

# Boîte à moustaches pour la distribution des précisions par langue
plt.subplot(1, 2, 2)
sns.boxplot(x='Language', y='Response_Accuracy', data=df, palette='viridis')
plt.title('Distribution des précisions par langue')
plt.xlabel('Langue')
plt.ylabel('Précision')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f'{save_path}/language_accuracy_analysis.png')
plt.close()
print(f'{save_path}/language_accuracy_analysis.png')

# 3. Tests statistiques
# ANOVA pour tester si les différences entre langues sont significatives
languages = df['Language'].unique()
if len(languages) > 1:  # S'assurer qu'il y a au moins 2 langues pour faire l'ANOVA
    groups = [df[df['Language'] == lang]['Response_Accuracy'].values for lang in languages]
    f_stat, p_value = stats.f_oneway(*groups)
    
    print("\nRésultats ANOVA:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Différence significative entre les langues (p<0.05)")
        
        # Analyse plus détaillée avec Tukey HSD pour voir quelles paires de langues diffèrent
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tukey = pairwise_tukeyhsd(df['Response_Accuracy'], df['Language'], alpha=0.05)
        print("\nTest post-hoc Tukey HSD:")
        print(tukey)
    else:
        print("Pas de différence significative entre les langues (p>0.05)")

# 4. Relation avec d'autres variables (optionnel)
print("\nMatrice de corrélation entre Response_Accuracy et d'autres variables numériques:")
correlation_columns = ['Response_Accuracy', 'Input_Text_Length', 'Response_Tokens', 
                       'User_Rating', 'User_Experience_Score', 'Session_Duration_sec', 'Response_Speed_sec']
correlation_columns = [col for col in correlation_columns if col in df.columns]
correlation_matrix = df[correlation_columns].corr()
print(correlation_matrix['Response_Accuracy'].sort_values(ascending=False))

# 5. Vérifier si la plateforme ou le modèle influence la relation entre langue et précision
if 'AI_Platform' in df.columns and 'AI_Model_Version' in df.columns:
    print("\nPrécision moyenne par langue et plateforme:")
    platform_lang_accuracy = df.groupby(['AI_Platform', 'Language'])['Response_Accuracy'].mean().unstack()
    print(platform_lang_accuracy)