import sys
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import pickle

model_preterm = pickle.load(open('basic_rf_early_preterm.save', 'rb'))

phylotype_1_test = pd.read_csv("input/phylotypes/phylotype_relabd.1e_1.csv")
metadata_test = pd.read_csv("input/metadata/metadata_normalized.csv")
diversity_test = pd.read_csv("input/alpha_diversity/alpha_diversity.csv")
valencia_test = pd.read_csv("input/community_state_types/cst_valencia_dummies.csv")

all_features_test = phylotype_1_test.merge(metadata_test, how='outer', on = 'specimen').merge(diversity_test, how='outer', on = 'specimen').merge(valencia_test, how='outer', on = 'specimen')

party_id = all_features_test['participant_id'].to_numpy()
specs = all_features_test['specimen'].to_numpy()

all_features_test_nospe = all_features_test.drop(columns=['participant_id','specimen', 'project','age'])
X = all_features_test_nospe.to_numpy()

specimen_participant = {
    sp: p
    for (sp, p) in zip(
        all_features_test.specimen,
        all_features_test.participant_id
    )
}

was_preterm = []
participant = []
probability = []

predicts = pd.DataFrame()
was_preterm = model_preterm.predict(X)
probability = model_preterm.predict_proba(X)[:,1]
for i in range(len(X)):
    participant.append(specimen_participant[specs[i]])

predicts['was_early_preterm'] = was_preterm.astype(int)
predicts['participant'] = participant
predicts['probability'] = probability

predicts_by_participant = predicts.groupby('participant').max()

predicts_by_participant.to_csv("output/predictions.csv")
