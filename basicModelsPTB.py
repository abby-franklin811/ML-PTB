import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

phylotype_1 = pd.read_csv("phylotypes/phylotype_relabd.1e_1.csv")
metadata = pd.read_csv("metadata/metadata_normalized.csv")
diversity = pd.read_csv("alpha_diversity/alpha_diversity.csv")
valencia = pd.read_csv("community_state_types/cst_valencia_dummies.csv")
answers = pd.read_csv("metadata/metadata.csv")

Ys = answers[['participant_id', 'was_preterm', 'was_early_preterm']].copy()

all_features = phylotype_1.merge(metadata, how='outer', on = 'specimen').merge(diversity, how='outer', on = 'specimen').merge(valencia, how='outer', on = 'specimen')

all_features = all_features.drop(columns=['participant_id','specimen','project','age'])

Y = Ys['was_preterm']
Y_early = Ys['was_early_preterm']
X = all_features.to_numpy()

def RFPipeline(X, Y):
    clf = RandomForestClassifier(random_state=42).fit(X,Y) #running model
    return clf
    

import pickle
model_preterm = RFPipeline(X,Y)
model_early_preterm= RFPipeline(X, Y_early)
pickle.dump(model_preterm, open('basic_rf_preterm.save', 'wb'))
pickle.dump(model_early_preterm, open('basic_rf_early_preterm.save','wb'))