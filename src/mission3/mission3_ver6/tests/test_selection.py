from unittest.mock import patch

import numpy as np
import pytest
from sklearn.metrics import f1_score

from mission3_ver6 import selection


def test_thresholds_fit_only_other_groups_and_each_stop_row_scored_once():
    targets = (np.arange(18)[:,None]+np.arange(9)[None,:])%3 == 0
    probabilities = np.random.default_rng(42).uniform(.05,.95,targets.shape)
    groups = np.repeat(np.arange(9),2)
    selector = selection.StopSelector(targets,groups,seed=3,n_splits=3)
    original = selection._best_threshold
    with patch.object(selection,"_best_threshold",wraps=original) as mocked:
        report = selector.score(probabilities)
    assert len(mocked.call_args_list)==27
    predictions=np.zeros_like(targets)
    seen=np.zeros(len(targets),int)
    for fold,(fit,test) in enumerate(selector.folds):
        assert not set(groups[fit])&set(groups[test])
        seen[test]+=1
        for label in range(9):
            args=mocked.call_args_list[fold*9+label].args
            np.testing.assert_array_equal(args[0],targets[fit,label])
            np.testing.assert_array_equal(args[1],probabilities[fit,label])
        predictions[test]=probabilities[test]>=np.array(report["thresholds_by_fold"][fold])
    assert np.all(seen==1)
    assert report["stop_macro_f1"]==f1_score(targets,predictions,average="macro",zero_division=0)


def test_selection_f1_primary_ap_only_ties_and_bad_inputs_rejected():
    assert selection.better({"stop_macro_f1":.6,"stop_macro_ap":.1},{"stop_macro_f1":.59,"stop_macro_ap":.9})
    assert not selection.better({"stop_macro_f1":.59,"stop_macro_ap":.9},{"stop_macro_f1":.6,"stop_macro_ap":.1})
    assert selection.better({"stop_macro_f1":.6,"stop_macro_ap":.8},{"stop_macro_f1":.6,"stop_macro_ap":.7})
    selector=selection.StopSelector(np.array([[1,0],[0,1],[1,1]]),[0,1,2])
    with pytest.raises(ValueError,match="finite"):
        selector.score(np.full((3,2),np.nan))
