import numpy as np
import glob
from astropy.io import fits

# 1. Search for ANY version of the file (using the * wildcard)
# Replace 'bn120123456' with your specific burst ID
def find_files(search_pattern):
    #search_pattern = f'{fermi_id}/glg_tcat_all_{fermi_id}_v*.fit'
    found_files = glob.glob(search_pattern)

    # 2. Check if we found anything
    if found_files:
        # 3. Sort the list. 
        # Since "v01" comes alphabetically after "v00", the last item is the newest.
        found_files.sort()
        best_file = found_files[-1]
        
        print(f"Using the latest version: {best_file}")
        return best_file 
        # Now you can open 'best_file' to check the detectors
    else:
        #print("No file found!")
        return Exception("No file found!")

