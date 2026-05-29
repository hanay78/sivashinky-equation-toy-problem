import numpy as np
import sys
from siva import simular_ks_sts
from unittest.mock import patch

# Mock animar_sivashinsky to avoid GUI/plots during test
with patch('siva.animar_sivashinsky') as mock_anim:
    print("Testing RK4 method for T_max=0.2...")
    try:
        simular_ks_sts(metodo='rk4', T_max=0.2)
        print("\nRK4 test completed successfully.")
    except Exception as e:
        print(f"\nRK4 test failed: {e}")
        sys.exit(1)

with patch('siva.animar_sivashinsky') as mock_anim:
    print("\nTesting STS method for T_max=0.2...")
    try:
        simular_ks_sts(metodo='sts', T_max=0.2)
        print("\nSTS test completed successfully.")
    except Exception as e:
        print(f"\nSTS test failed: {e}")
        sys.exit(1)
