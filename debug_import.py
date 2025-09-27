import os
os.environ['YOLO_VERBOSE'] = 'False'

print("--- Starting Import Debug ---")
try:
    print("Attempting to import CourtPointsTrainer...")
    from courtpoints.trainer import CourtPointsTrainer
    print("✅ Import successful!")
except Exception as e:
    import traceback
    print(f"❌ Import failed!")
    print(f"   Error Type: {type(e).__name__}")
    print(f"   Error: {e}")
    print("--- Traceback ---")
    traceback.print_exc()

print("--- Import Debug Finished ---")
