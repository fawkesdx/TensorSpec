import sys
import traceback

print("Current Python environment:", sys.executable)

try:
    import chinook
    print("SUCCESS: Chinook is installed at:", chinook.__file__)
    
    # Try importing the specific module we use
    import chinook.build_lib as build_lib
    print("SUCCESS: chinook.build_lib imported correctly.")
    
except Exception as e:
    print("\nFAILED TO IMPORT CHINOOK. Here is the real error:")
    print("-" * 50)
    traceback.print_exc()
    print("-" * 50)