import pandas as pd

df = pd.read_csv('compoverrelowTm_Ivan.csv')
print(df[df['Tm'] < 1200].head(20))