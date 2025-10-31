#!/usr/bin/env python
"""Quick test to verify imports are working"""

try:
    from backend.enhanced_data_collector import EnhancedDataCollector
    from backend.holistic_detector import HolisticDetector
    from backend.unified_classifier import UnifiedClassifier
    print("✅ All imports successful!")
    print("✅ The import errors have been fixed!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
