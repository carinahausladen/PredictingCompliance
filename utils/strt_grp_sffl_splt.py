import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def str_grp_splt(df,
                 grp_col_name,
                 y_col_name,
                 train_share=0.8,
                 minority_label=1):
    """

    Args:
        df (pandas.DataFrame):
        grp_col_name (str, optional): Define grps which should not appear in both: train and test.
        y_col_name (str, optional): Define label of which distribution in train and test should be similar to whole df.
        train_share (float, optional): Defaults set to 80 percent.

    Returns:
        X_train, X_test, y_train, y_test (indices)
    """

    train_size = df.shape[0] * train_share
    lbl_cnts = df.groupby([y_col_name]).size().sort_values()

    # define which groups to select for minority label
    mnrty_lbl_cnt = min(lbl_cnts)
    mnrty_lbl = lbl_cnts.index[0]

    share_minority = mnrty_lbl_cnt / sum(lbl_cnts)
    train_minority = train_size * share_minority  # number of observations that should reflect the minority (1) class in the training sample

    cnt_grp = df.loc[df[y_col_name] == mnrty_lbl, [grp_col_name, y_col_name]].groupby(
        [grp_col_name, y_col_name]).size()

    cnt_grp = cnt_grp.sort_values(ascending=False)
    min_idx = np.absolute(cnt_grp.cumsum() - train_minority).argmin()
    train_grp_mnrty = list(cnt_grp.index.get_level_values(level=0)[:min_idx + 1])
    test_grp_mnrty = list(set(cnt_grp.index.get_level_values(level=0)) - set(train_grp_mnrty))

    # for two/three labels
    train_grp_medium = []
    test_grp_medium = []
    if len(lbl_cnts.index) > 2:
        # define which groups to select for medium label
        mdium_lbl_cnt = lbl_cnts[1]  # sorting ensures that 1 is the medium label (only if there are 3 labels!)
        mdium_lbl = lbl_cnts.index[1]

        share_medium = mdium_lbl_cnt / sum(lbl_cnts)
        test_medium = train_size * share_medium

        cnt_grp = df.loc[df[y_col_name] == mdium_lbl, [grp_col_name, y_col_name]].groupby(
            [grp_col_name, y_col_name]).size()
        cnt_grp = cnt_grp.sort_values(ascending=False)
        cnt_grp = cnt_grp.reset_index()
        cnt_grp = cnt_grp.loc[~cnt_grp[grp_col_name].isin(train_grp_mnrty + test_grp_mnrty)]  # exclude groups which are already taken
        cnt_grp = cnt_grp.set_index([grp_col_name, y_col_name])[0]  # indexer is needed to convert df back to series!

        min_idx = np.absolute(cnt_grp.cumsum() - test_medium).argmin()
        train_grp_medium = list(cnt_grp.index.get_level_values(level=0)[:min_idx + 1])
        test_grp_medium = list(set(cnt_grp.index.get_level_values(level=0)) - set(train_grp_medium))

    # define which groups to select for majority label
    mjrty_lbl_cnt = max(lbl_cnts)  # just for testing
    mjrty_lbl = lbl_cnts.idxmax()

    df_1 = df.loc[df[y_col_name] == mjrty_lbl, [grp_col_name, y_col_name]]  # only grps with majority label
    df_1 = df_1.loc[~df_1[grp_col_name].isin(
        train_grp_mnrty + test_grp_mnrty + train_grp_medium + test_grp_medium)]  # exclude taken grps
    gss = GroupShuffleSplit(n_splits=1, train_size=train_share, random_state=42)

    train_idx_mjrty, test_idx_mjrty = \
        [(_a, _b) for _a, _b in gss.split(None, df_1[y_col_name], groups=df_1[grp_col_name])][0]
    train_group_mjrty = list(np.unique(df_1.iloc[train_idx_mjrty, :][grp_col_name].values))
    test_group_mjrty = list(np.unique(df_1.iloc[test_idx_mjrty, :][grp_col_name].values))

    # do split based on groups (already include the strata information!)
    train_grp = train_grp_mnrty + train_grp_medium + train_group_mjrty
    test_grp = test_grp_mnrty + test_grp_medium + test_group_mjrty
    list(set(train_grp) & set(test_grp))  # there should be no overlapping groups

    train_idx = df.loc[df[grp_col_name].isin(train_grp)].index
    test_idx = df.loc[df[grp_col_name].isin(test_grp)].index

    return train_idx, test_idx

#train_idx, test_idx = str_grp_splt(df,
#                                   grp_col_name=grp_col_name,
#                                   y_col_name=y_col_name,
#                                   train_share=0.8)