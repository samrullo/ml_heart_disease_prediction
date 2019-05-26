import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import namedtuple


def load_hear_disease_data(samples_file, targets_file):
    features_df = pd.read_csv(samples_file)
    targets_df = pd.read_csv(targets_file)
    mrg_df = pd.merge(left=features_df, right=targets_df, on='patient_id')
    features = ['slope_of_peak_exercise_st_segment', 'resting_blood_pressure',
                'chest_pain_type', 'num_major_vessels',
                'fasting_blood_sugar_gt_120_mg_per_dl', 'resting_ekg_results',
                'serum_cholesterol_mg_per_dl', 'oldpeak_eq_st_depression', 'sex', 'age',
                'max_heart_rate_achieved', 'exercise_induced_angina']

    label_col = ['heart_disease_present']

    X = mrg_df[features].as_matrix()
    y = mrg_df[label_col].as_matrix()
    std_scaler = StandardScaler()
    X = std_scaler.fit_transform(X)
    y = y.flatten()
    ret = namedtuple('ret', ['samples', 'targets', 'feature_labels'])
    ret.samples = X
    ret.targets = y
    ret.feature_labels = features
    return ret
