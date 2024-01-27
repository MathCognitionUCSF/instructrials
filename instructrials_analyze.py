#%%
from instructrials.simulate import generate_data
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

#%%
df = generate_data(100)
# %%
df['delta'] = df['post_test'] - df['pre_test']
df.head()
#%% Pearson correlation
correlation, p_value = pearsonr(df['delta'], df['phonology_task_1'])
# Print the results
print(f"Correlation coefficient: {correlation}")
print(f"P-value: {p_value}")

#%% Linear regression with random factor
model = smf.mixedlm("delta ~ phonology_task_1", df, groups=df["pidn"])
result = model.fit()
print(result.summary())

# %%  Scatter plot with a regression line
plot = sns.lmplot(x='phonology_task_1', y='delta', data=df, aspect=1.2, height=10,scatter_kws={"s": 100})

# Enhancing the plot
plt.title('Intervention vs Phonology', size=50)
plt.xlabel('Phonology', size=30)
plt.ylabel('Intervention effect: (post-pre test)', size=30)
plot.ax.tick_params(axis='both', which='major', labelsize=30)

# %% Plot distribution of phonology_task_1
sns.displot(df['phonology_task_1'], kde=True)

#%% Create 2 phonology groups based on a median split
df['phonology_task_1_group'] = df['phonology_task_1'] >= df['phonology_task_1'].median()
df['phonology_task_1_group'] = df['phonology_task_1_group'].replace({True: 'High', False: 'Low'})

#%% Boxplot of phonology_task_1_group vs delta
# Boxplot of phonology_task_1_group vs delta
plot = sns.boxplot(x='phonology_task_1_group', y='delta', data=df)
plt.title('Intervention vs Phonology', size=20)
plt.xlabel('Phonology', size=20)
plt.ylabel('Intervention effect: (post-pre test)', size=20)
plot.tick_params(axis='both', which='major', labelsize=20)
#%% Run mixed effects model with delta as dependent, phonology groups as independent and pidn as random factor
model = smf.mixedlm("delta ~ phonology_task_1_group", df, groups=df["pidn"])
result = model.fit()
print(result.summary())
# %%
