"""
Run MIQP examples
"""
from __future__ import print_function

import examples.power_converter.run_example as power_converter
import examples.random_miqp.run_example as random_miqp


if __name__ == "__main__":

    random_miqp.run_example()
    power_converter.run_example()
