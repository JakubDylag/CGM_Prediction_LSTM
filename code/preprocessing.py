import numpy
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

def get_df_all(folder, num_groups, df_pt_groups_cols):
    """
    :param folder: Data folder path
    :param num_groups: Number of patient groups for GMM Clustering
    :return: Data frames for CGM and Patient data
    """
    path_cgm = folder + '/DeviceCGM.txt'
    path_pt_medical = folder + '/MedicalCondition.txt'
    path_pt_phys = folder + '/DiabPhysExam.txt'
    path_pt_screen = folder + '/DiabScreening.txt'
    path_pt_soc_eco = folder + '/DiabSocioEcon.txt'
    df_cgm, df_pt_medical, df_pt_phys, df_pt_screen, df_pt_soc_eco = fetch(path_cgm, path_pt_medical, path_pt_phys,
                                                                           path_pt_screen, path_pt_soc_eco)

    df_cgm, df_pt_hba1c = process_cgm(df_cgm)
    # df_pt_medical = process_medical(df_pt_medical)
    df_pt_phys = process_pt_phys(df_pt_phys)
    df_pt_screen = process_pt_screen(df_pt_screen)
    df_pt_soc_eco = process_pt_soc_eco(df_pt_soc_eco)

    # Merge Patient Details into single df
    df_pt_all = df_pt_phys.merge(df_pt_screen, on="PtID")
    df_pt_all = df_pt_all.merge(df_pt_soc_eco, on="PtID")
    df_pt_all = df_pt_hba1c.merge(df_pt_all, on="PtID")
    df_pt_all = df_pt_all.set_index("PtID")

    # calculate BMI
    df_pt_all["BMI"] = (df_pt_all["Weight"] / (df_pt_all["Height"] * df_pt_all["Height"]))
    df_pt_all["BMI"] = df_pt_all["BMI"] * (10000)

    if (num_groups == 1):
        return df_cgm, df_pt_all
    else:
        # Cluster
        df_pt_groups_in = df_pt_all[df_pt_groups_cols]
        gmm = GaussianMixture(n_components=num_groups, covariance_type='full', max_iter=200,
                              n_init=20)  # https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        gmm.fit(df_pt_groups_in)
        df_pt_all["Group"] = gmm.predict(df_pt_groups_in)

        # Group patients by Cluster
        df_pt_all_groups = df_pt_all.groupby("Group")
        return df_cgm, df_pt_all_groups


def create_time_series(pt_ids, df_cgm, series_min_len):
    """
    :param pt_ids: list of patient IDs to be included
    :param df_cgm: Pandas Dataframe of all cgm measurements
    :param series_min_len: Minimum length of time series
    :param test_size:
    :return: List of
    """
    max_time_diff = 15 #mins
    shift_by = 1
    
    group_cgm = df_cgm[df_cgm["PtID"].isin(pt_ids)]
    group_cgm_pts = group_cgm.groupby("PtID")

    out_all = []

    for name, group_cgm_pt in group_cgm_pts: #For every patient
        group_cgm_pt = group_cgm_pt.sort_values("DeviceDtTm")
        group_cgm_pt["Diff"] = group_cgm_pt["DeviceDtTm"].diff()
        group_cgm_pt["Diff_threshold"] = group_cgm_pt["Diff"] > pd.Timedelta(max_time_diff, 'm')
        group_cgm_pt["Sum"] = group_cgm_pt["Diff_threshold"].cumsum()
        group_cgm_pt_series = group_cgm_pt.groupby("Sum")

        for name, series in group_cgm_pt_series: #For every
            values = series["Value"].tolist()  # .to_numpy()
            if len(values) >= series_min_len:
                # Split into chunks of size "max_time_diff", shifted along by "shift_by"
                for offset in range(0, len(values) - series_min_len + 1, shift_by):
                    # print(offset, offset+series_min_len+1, len(series[offset:offset+series_min_len]))
                    # np.append(out_series, series[offset:offset + series_min_len])
                    out_all.append(values[offset:offset+series_min_len])

    return out_all


def fetch(path_cgm, path_pt_medical, path_pt_phys, path_pt_screen, path_pt_soc_eco):
    df_cgm = pd.read_csv(path_cgm, sep='|', encoding='utf-16')
    df_pt_medical = pd.read_csv(path_pt_medical, sep='|', encoding='utf-16')
    df_pt_phys = pd.read_csv(path_pt_phys, sep='|', encoding='utf-16')
    df_pt_screen = pd.read_csv(path_pt_screen, sep='|', encoding='utf-16')
    df_pt_soc_eco = pd.read_csv(path_pt_soc_eco, sep='|', encoding='utf-16')
    return df_cgm, df_pt_medical, df_pt_phys, df_pt_screen, df_pt_soc_eco


def process_cgm(df_cgm):
    df_cgm["DeviceDtTm"] = pd.to_datetime(df_cgm["DeviceDtTm"], yearfirst=True)
    # TODO: remove RecordType == Calibration?

    # Calculate HbA1c (mean glucose level)
    df_pt_hba1c = df_cgm.groupby("PtID")["Value"].mean()
    df_pt_hba1c = df_pt_hba1c.to_frame()
    df_pt_hba1c = df_pt_hba1c.rename(columns={"Value": "HbA1c"})
    return df_cgm, df_pt_hba1c


def process_pt_medical(df_pt_medical):
    return df_pt_medical


def process_pt_phys(df_pt_phys):
    mask = (df_pt_phys["WeightUnits"] == 'lbs')
    df_temp = df_pt_phys[mask]
    df_pt_phys.loc[mask, 'Weight'] = df_temp["Weight"] * 0.453592

    # in to cm
    mask = (df_pt_phys["HeightUnits"] == 'in')
    df_temp = df_pt_phys[mask]
    df_pt_phys.loc[mask, 'Height'] = df_temp["Height"] * 2.54

    # PEAbnormal to 1 0
    df_pt_phys["PEAbnormal"] = df_pt_phys["PEAbnormal"].replace({"Yes": 1, "No": 0})

    # Take mean or max of all readings
    df_pt_phys = df_pt_phys[["PtID", "Weight", "Height", "PEHeartRt", "PEAbnormal"]]
    df_pt_phys_mean = df_pt_phys[["PtID", "Weight", "Height", "PEHeartRt"]].groupby("PtID").mean()
    df_pt_phys_max = df_pt_phys[["PtID", "PEAbnormal"]].groupby("PtID").max()
    df_pt_phys = df_pt_phys_mean.join(df_pt_phys_max)
    return df_pt_phys


def process_pt_screen(df_pt_screen):
    # Sex to 1 0
    df_pt_screen["Sex"] = df_pt_screen["Sex"].replace({"M": 1, "F": 0})
    # Race One Hot encoding
    df_pt_screen["Race"] = df_pt_screen["Race"].replace(
        {'Black/African American': 'Black', 'More than one race': "Mixed", 'Unknown/not reported': "Unknown",
         'American Indian/Alaskan Native': "Native"})
    df_onehot = pd.get_dummies(df_pt_screen["Race"])
    df_pt_screen = df_pt_screen.join(df_onehot)
    # Select Cols
    df_pt_screen = df_pt_screen[["PtID", "Sex", 'Black', 'White', 'Asian', 'Mixed', 'Unknown', 'Native']]
    return df_pt_screen


def process_pt_soc_eco(df_pt_soc_eco):
    # EducationLevel to Years in Educations
    df_pt_soc_eco["EducationLevel"] = df_pt_soc_eco["EducationLevel"].replace({
        "Master's Degree (MA, MS, MSW, MBA, MPH)": 16,
        "Bachelor's Degree (BS,BA,AB)": 15,
        'Professional Degree (MD, DDS, DVM, LLB, JD)': 15,
        'Associate Degree (AA)': 14,
        'Some college but no degree': 13,
        'High school graduate/diploma/GED': 12,
        '12th grade - no diploma': 11.5,
        '11th grade': 11,
        '10th grade': 10,
        '9th grade': 9,
        '7th or 8th grade': 7.5,
        '1st, 2nd, 3rd, or 4th grade': 2.5,
        'Unknown': 0,
        'Does not wish to provide': 0,
    })
    # AnnualIncome to Int Thousands
    df_pt_soc_eco["AnnualIncome"] = df_pt_soc_eco["AnnualIncome"].replace({
        'Unknown': 0,
        'Does not wish to provide': 0,
        'Less than $25,000': 12.5,
        '$25,000 to less than $35,000': 30,
        '$35,000 to less than $50,000': 42.5,
        '$50,000 to less than $75,000': 62.5,
        '$75,000 to less than $100,000': 87.5,
        '$100,000 to less than $200,000': 150,
        '$200,000 or more': 200
    })
    df_pt_soc_eco = df_pt_soc_eco[["PtID", "EducationLevel", "AnnualIncome"]]
    return df_pt_soc_eco
