import sys
import pandas as pd
import pickle

model_preterm = pickle.load(open('basic_svm_gan_preterm.save', 'rb'))

phylotype_1_test = pd.read_csv("input/phylotypes/phylotype_relabd.1e_1.csv")
metadata_test = pd.read_csv("input/metadata/metadata_normalized.csv")

all_features = phylotype_1_test.merge(metadata_test, how='outer', on = 'specimen')

# all_features_test['participant_id'] = metadata_test['participant_id']
all_features_test = all_features.drop(columns=['collect_wk','Race: American Indian or Alaska Native', "Race: Asian", "Race: Black or African American", "Race: Native Hawaiian or Other Pacific Islander",
                           "Race: Unknown", "Race: White", "Ethnicity: Hispanic or Latino", "Ethnicity: Unknown"])

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

predicts['was_preterm'] = was_preterm.astype(int)
predicts['participant'] = participant
predicts['probability'] = probability

predicts_by_participant = predicts.groupby('participant').max()

predicts_by_participant.to_csv("output/predictions.csv")
