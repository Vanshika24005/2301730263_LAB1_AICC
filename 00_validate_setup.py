"""Task 1: validate that the environment has every required package."""
import sys
import numpy, pandas, sklearn, matplotlib, seaborn

print("Python      :", sys.version.split()[0])
for m in (numpy, pandas, sklearn, matplotlib, seaborn):
    print(f"{m.__name__:<12}:", m.__version__)
print("Setup OK - all required packages imported successfully.")
