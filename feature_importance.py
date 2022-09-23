model_preterm = pickle.load(open("basic_rf_preterm.save", 'rb'))
feature_names = [all_features.columns[i] for i in range(len(all_features.columns))]
importances = model_preterm.feature_importances_
forest_importances = pd.Series(importances, index=feature_names)
list(forest_importances.sort_values(ascending=False)[:200].index)

model_early_preterm = pickle.load(open("basic_rf_early_preterm.save", 'rb'))
feature_names = [all_features.columns[i] for i in range(len(all_features.columns))]
importances = model_early_preterm.feature_importances_
forest_importances = pd.Series(importances, index=feature_names)
list(forest_importances.sort_values(ascending=False)[:200].index)
