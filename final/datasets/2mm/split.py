import os
import pandas as pd

thisDir = os.getcwd()

df_combined=pd.read_csv(os.path.join(thisDir, 'combined_2mm.csv'))

df_bck = df_combined[df_combined['signal']==0]
df_sig = df_combined[df_combined['signal']==1] 

df_bck.to_csv(os.path.join(thisDir, 'background_2mm.csv'), index=False)
df_sig.to_csv(os.path.join(thisDir, 'signal_2mm.csv'), index=False)

