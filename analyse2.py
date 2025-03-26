import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import calendar
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# Charger les données
df = pd.read_csv('./archive/deepseek_vs_chatgpt.csv')

# Convertir la colonne Date en datetime
df['Date'] = pd.to_datetime(df['Date'])

# 1. Analyse comparative des plateformes AI
print("=== Analyse comparative des plateformes AI ===")
platform_metrics = df.groupby('AI_Platform').agg({
    'Response_Accuracy': 'mean',
    'Response_Speed_sec': 'mean',
    'User_Rating': 'mean',
    'User_Experience_Score': 'mean',
    'Correction_Needed': 'mean',
    'Customer_Support_Interactions': 'mean',
    'Active_Users': 'mean',
    'New_Users': 'mean',
    'Churned_Users': 'mean',
    'User_ID': 'count'  # Nombre d'observations
}).rename(columns={'User_ID': 'Observations'})

print(platform_metrics)

# Visualisation des métriques par plateforme
plt.figure(figsize=(14, 10))

# Normaliser les données pour une visualisation équilibrée
normalized_metrics = platform_metrics.copy()
for column in normalized_metrics.columns:
    normalized_metrics[column] = (normalized_metrics[column] - normalized_metrics[column].min()) / \
                                (normalized_metrics[column].max() - normalized_metrics[column].min())


def radar_chart(ax, metrics, platform_name):
    # Nombre de variables
    categories = metrics.columns
    N = len(categories)
    
    # Angle pour chaque variable
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Fermer le polygone
    
    # Valeurs pour chaque variable
    values = metrics.iloc[0].values.tolist()
    values += values[:1]  # Fermer le polygone
    
    # Dessiner le polygone
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=platform_name)
    ax.fill(angles, values, alpha=0.25)
    
    # Ajouter les labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8)
    
    # Ajouter les lignes de grille
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], size=8)
    ax.set_ylim(0, 1)
    
    # Ajouter un titre
    ax.set_title(platform_name, size=11, y=1.1)

platforms = normalized_metrics.index.tolist()
nrows = int(np.ceil(len(platforms) / 2))
fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(14, 5*nrows), subplot_kw=dict(projection='polar'))
axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

for i, platform in enumerate(platforms):
    radar_chart(axes[i], normalized_metrics.loc[[platform]], platform)

# Add this to hide empty subplots if number of platforms is odd
if len(platforms) % 2 != 0:
    axes[-1].remove()

plt.tight_layout()
plt.savefig(f'/plots/platform_comparison_radar.png')
plt.close()
print(f'/plots/platform_comparison_radar.png')

# 2. Analyse de l'expérience utilisateur
print("\n=== Analyse des facteurs influençant l'expérience utilisateur ===")

# Calculer les corrélations avec User_Rating et User_Experience_Score
correlation_cols = ['Response_Accuracy', 'Response_Speed_sec', 'Session_Duration_sec', 
                    'Input_Text_Length', 'Response_Tokens', 'Correction_Needed']
user_exp_corr = df[correlation_cols + ['User_Rating', 'User_Experience_Score']].corr()
print("Corrélations avec User_Rating:")
print(user_exp_corr['User_Rating'].sort_values(ascending=False))
print("\nCorrélations avec User_Experience_Score:")
print(user_exp_corr['User_Experience_Score'].sort_values(ascending=False))

# Visualisation des corrélations
plt.figure(figsize=(12, 10))
sns.heatmap(user_exp_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Matrice de corrélation des facteurs liés à l\'expérience utilisateur')
plt.tight_layout()
plt.savefig(f'/plots/user_experience_correlation.png')
plt.show()

# Analyse par Topic_Category
topic_experience = df.groupby('Topic_Category').agg({
    'User_Rating': 'mean',
    'User_Experience_Score': 'mean',
    'Response_Accuracy': 'mean',
    'Response_Speed_sec': 'mean',
    'Correction_Needed': 'mean',
    'User_ID': 'count'
}).rename(columns={'User_ID': 'Query_Count'}).sort_values(by='User_Rating', ascending=False)

print("\nExpérience utilisateur par catégorie de sujet:")
print(topic_experience)

plt.figure(figsize=(12, 6))
sns.barplot(x=topic_experience.index, y=topic_experience['User_Rating'], palette='viridis')
plt.title('Satisfaction utilisateur moyenne par catégorie de sujet')
plt.xlabel('Catégorie de sujet')
plt.ylabel('Évaluation moyenne')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'/plots/topic_user_rating.png')
plt.show()

# 3. Analyse des tendances temporelles
print("\n=== Analyse des tendances temporelles ===")

# Regrouper par date et calculer les métriques moyennes
daily_metrics = df.groupby('Date').agg({
    'Active_Users': 'mean',
    'New_Users': 'mean',
    'Churned_Users': 'mean',
    'Daily_Churn_Rate': 'mean',
    'Retention_Rate': 'mean',
    'Response_Accuracy': 'mean',
    'User_Rating': 'mean',
    'User_Experience_Score': 'mean',
    'Response_Speed_sec': 'mean'
})

print("Métriques quotidiennes:")
print(daily_metrics)

# Visualisation des tendances temporelles
plt.figure(figsize=(14, 10))

# Utilisateurs actifs, nouveaux et perdus
plt.subplot(3, 1, 1)
daily_metrics[['Active_Users', 'New_Users', 'Churned_Users']].plot(ax=plt.gca())
plt.title('Évolution des utilisateurs au fil du temps')
plt.ylabel('Nombre d\'utilisateurs')
plt.grid(True, linestyle='--', alpha=0.7)

# Taux de rétention et de churn
plt.subplot(3, 1, 2)
daily_metrics[['Retention_Rate', 'Daily_Churn_Rate']].plot(ax=plt.gca())
plt.title('Évolution des taux de rétention et de churn')
plt.ylabel('Taux')
plt.grid(True, linestyle='--', alpha=0.7)

# Précision et satisfaction
plt.subplot(3, 1, 3)
daily_metrics[['Response_Accuracy', 'User_Rating', 'User_Experience_Score']].plot(ax=plt.gca())
plt.title('Évolution de la précision et de la satisfaction')
plt.ylabel('Score')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(f'/plots/temporal_trends.png')
plt.show()

# 4. Analyse par type d'appareil
device_metrics = df.groupby('Device_Type').agg({
    'Response_Accuracy': 'mean',
    'Response_Speed_sec': 'mean',
    'User_Rating': 'mean',
    'User_Experience_Score': 'mean',
    'Session_Duration_sec': 'mean',
    'Correction_Needed': 'mean',
    'User_ID': 'count'
}).rename(columns={'User_ID': 'Query_Count'})

print("\n=== Analyse par type d'appareil ===")
print(device_metrics)

plt.figure(figsize=(14, 6))
metrics_to_plot = ['Response_Accuracy', 'User_Rating', 'User_Experience_Score']
device_metrics[metrics_to_plot].plot(kind='bar', ax=plt.gca())
plt.title('Métriques de performance par type d\'appareil')
plt.ylabel('Score moyen')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Métrique')
plt.tight_layout()
plt.savefig(f'/plots/device_performance.png')
plt.show()

# 5. Analyse de la fidélité des utilisateurs
print("\n=== Analyse de la fidélité des utilisateurs ===")
user_return = df.groupby('User_Return_Frequency').agg({
    'User_ID': 'count',
    'User_Rating': 'mean',
    'User_Experience_Score': 'mean',
    'Response_Accuracy': 'mean',
    'Response_Speed_sec': 'mean',
    'Correction_Needed': 'mean'
}).rename(columns={'User_ID': 'Query_Count'})

print(user_return)

# Visualisation de la relation entre fréquence de retour et satisfaction
plt.figure(figsize=(10, 6))
sns.scatterplot(x='User_Return_Frequency', y='User_Rating', 
                size='Query_Count', sizes=(50, 500), 
                data=user_return.reset_index(), alpha=0.7)
plt.title('Relation entre fréquence de retour et satisfaction utilisateur')
plt.xlabel('Fréquence de retour')
plt.ylabel('Évaluation moyenne')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'/plots/return_frequency_satisfaction.png')
plt.show()