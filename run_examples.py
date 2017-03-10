"""
Run MIQP examples
"""
from __future__ import print_function

import examples.vehicle.run_example as vehicle
import examples.power_converter.run_example as power_converter
import examples.random_qp.run_example as random_qp



if __name__ == "__main__":

    # vehicle.run_example()

    random_qp.run_example()
    power_converter.run_example()
