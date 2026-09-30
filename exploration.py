import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pathlib import Path

folder = Path(__file__).resolve().parent / "results" / "figures"
folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/processed/cohort.csv")

#print(df.head())
#print(df.describe())
#print(df.isna().sum())

plt.plot()
plt.hist(df['BMXBMI'].dropna(), bins=30, edgecolor='black')
plt.title('Distribution of BMI')
plt.legend(['BMI'])
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.savefig(folder / "bmi_distribution.png" , dpi=300, bbox_inches='tight')
plt.show()

plt.plot()
plt.hist(df['RIDAGEYR'].dropna(), bins=30, edgecolor='black')
plt.title('Distribution of Age')
plt.legend(['Age'])
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig(folder / "age_distribution.png" , dpi=300, bbox_inches='tight')
plt.show()


"""Effectifs par sexe"""

df_sex = df.groupby('RIAGENDR').size().reset_index(name='count')
plt.bar(df_sex['RIAGENDR'], df_sex['count'], color=['blue', 'red'])
plt.title('Effectifs par sexe')
plt.xlabel('Sexe')
plt.ylabel('Effectifs')
plt.legend(handles=[
    Patch(facecolor="blue", label="Hommes"),
    Patch(facecolor="red", label="Femmes"),
])
plt.xticks([1, 2], ["Hommes", "Femmes"])
plt.savefig(folder / "sex_distribution.png" , dpi=300, bbox_inches='tight')
plt.show()


#boxplots d'age par event_5y

age_0 = df.loc[df["event_5y"] == 0, "RIDAGEYR"]
age_1 = df.loc[df["event_5y"] == 1, "RIDAGEYR"]

plt.figure()
plt.boxplot(
    [age_0, age_1],
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", color="blue"),
)
plt.xticks([1, 2], ["Pas de décès à 5 ans", "Décès à 5 ans"])
plt.ylabel("Âge à l’inclusion")
plt.title("Âge selon le statut de décès à 5 ans")
plt.savefig(folder / "age_by_event_5y.png" , dpi=300, bbox_inches='tight')
plt.show()

#boxplots of bmi by event_5y

bmi_0 = df.loc[df["event_5y"] == 0, "BMXBMI"]
bmi_1 = df.loc[df["event_5y"] == 1, "BMXBMI"]

plt.figure()
plt.boxplot(
    [bmi_0, bmi_1],
    patch_artist=True,
    boxprops=dict(facecolor="lightgreen", color="green"),
)
plt.xticks([1, 2], ["Pas de décès à 5 ans", "Décès à 5 ans"])
plt.ylabel("IMC à l’inclusion")
plt.title("IMC selon le statut de décès à 5 ans")
plt.savefig(folder / "bmi_by_event_5y.png" , dpi=300, bbox_inches='tight')
plt.show()