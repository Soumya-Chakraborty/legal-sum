import numpy as np
import vsum_tools

def test_metrics_upgrade():
    ypred = np.array([0.1, 0.9, 0.8, 0.2, 0.9])
    cps = np.array([[0, 9], [10, 19], [20, 29], [30, 39], [40, 49]])
    nfps = [10, 10, 10, 10, 10]
    positions = np.array([0, 10, 20, 30, 40])
    
    # Test dynamic proportion
    summary = vsum_tools.generate_summary(ypred, cps, 50, nfps, positions, proportion=-1)
    assert summary.shape == (50,)
    
    # Test multiple F1 metrics
    user_summary = np.ones((2, 50))
    user_summary[0, 30:] = 0
    f_avg, f_max, prec, rec = vsum_tools.evaluate_summary(summary, user_summary, 'all')
    assert f_max >= f_avg
