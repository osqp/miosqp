class Results(object):
    """
    MIOSQP Results
    """
    def __init__(self, x, upper_glob, run_time, status,
                 osqp_solve_time, osqp_iter_avg):
        self.x = x
        self.upper_glob = upper_glob
        self.run_time = run_time
        self.status = status
        self.osqp_solve_time = osqp_solve_time
        self.osqp_iter_avg = osqp_iter_avg
