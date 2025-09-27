import torch
import numpy as np
from pathlib import Path
import os

# Suppress verbose logging from ultralytics
os.environ['YOLO_VERBOSE'] = 'False'

from courtpoints import (
    CourtPointsModel,
    CourtPointsTrainer,
    CourtPointsValidator
)
from courtpoints.trainer import CourtPointsLoss
from courtpoints.validator import CourtPointsMetrics


def run_test(test_function, test_name):
    """Helper function to run a test and print its status."""
    print(f"\n--- Testing {test_name} ---")
    try:
        test_function()
        print(f"✅ {test_name}: PASS")
        return True
    except Exception as e:
        import traceback
        print(f"❌ {test_name}: FAIL")
        print(f"   Error: {e}")
        # print(traceback.format_exc())
        return False


def test_model_creation():
    """Test if the CourtPointsModel can be initialized successfully."""
    model = CourtPointsModel(nc=3, verbose=False)
    assert model is not None, "Model should not be None"
    assert isinstance(model.model[-1], torch.nn.Module), "Model head should be a torch Module"
    print("   Model created successfully.")
    print(f"   Final model head is: {type(model.model[-1]).__name__}")

def test_loss_function():
    """Test if the CourtPointsLoss can compute a loss value."""
    # 1. Create a dummy model
    print("   Creating a dummy CourtPointsModel...")
    model = CourtPointsModel(nc=3, verbose=False)

    # 2. Attach a mock 'args' object to the model, as the loss function needs it
    class MockArgs:
        point_loss_weight = 2.0
        class_loss_weight = 1.0
        conf_loss_weight = 1.0
    model.args = MockArgs()
    print("   Model and mock args created.")

    # 3. Initialize the loss function
    criterion = CourtPointsLoss(model)
    print(f"   Initialized {type(criterion).__name__} successfully.")

    # 4. Create dummy predictions and batch
    batch_size = 2
    num_preds = 8400 # A realistic number of predictions from the model
    num_gt = 3
    
    # Dummy model output (feats) - Use the 'no' attribute from the head for correct channel count
    head = model.model[-1]
    feats = [
        torch.randn(batch_size, head.no, 80, 80),
        torch.randn(batch_size, head.no, 40, 40),
        torch.randn(batch_size, head.no, 20, 20),
    ]

    # Dummy ground truth batch
    batch = {
        'batch_idx': torch.tensor([0, 0, 1]),
        'cls': torch.tensor([[0], [1], [2]]),
        'bboxes': torch.tensor([
            [0.5, 0.5, 0.02, 0.02], # GT point 1 in image 0
            [0.4, 0.4, 0.02, 0.02], # GT point 2 in image 0
            [0.6, 0.6, 0.02, 0.02], # GT point 3 in image 1
        ]),
        'img': torch.randn(batch_size, 3, 640, 640)
    }

    # 5. Compute loss
    return_value = criterion(feats, batch)
    print(f"   Criterion returned a {type(return_value)} of length {len(return_value)}.")
    loss, loss_items = return_value

    print(f"   Computed total loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be greater than zero"
    print(f"   Loss items: point={loss_items[0]:.4f}, class={loss_items[1]:.4f}, conf={loss_items[2]:.4f}")

def test_validation_metrics():
    """Test if the CourtPointsValidator and CourtPointsMetrics work correctly."""
    # 1. Initialize metrics
    metrics = CourtPointsMetrics(thresholds=[10, 20])
    print(f"   Initialized {type(metrics).__name__} successfully.")

    # 2. Create dummy matched stats
    # (pred_cls, gt_cls, distance)
    matched_stats = [
        [0, 0, 5.0],  # Correct match, distance 5px
        [1, 1, 8.0],  # Correct match, distance 8px
        [2, 0, 12.0], # Incorrect class, distance 12px
        [1, 1, 25.0], # Correct class, but distance 25px (too far)
    ]
    metrics.update(matched_stats)
    
    # 3. Set total number of ground truths and compute
    total_ground_truths = 5 # Let's say there were 5 GT points in total
    metrics.set_total_gt(total_ground_truths)
    results = metrics.compute()
    print(f"   Computed metrics: {results}")

    # 4. Verify results
    # PA@10px: 2 correct matches (dist 5 and 8) / 5 total GTs = 0.4
    # PA@20px: 3 correct matches (dist 5, 8, 12 - but class is wrong for 12) -> only 2 are TP
    # Let's re-check the logic. A TP is correct class AND within distance.
    # For PA@10px: TP = (cls==cls & dist<=10). Matches are [0,0,5] and [1,1,8]. So 2 TPs. Recall = 2/5 = 0.4
    # For PA@20px: TP = (cls==cls & dist<=20). Matches are [0,0,5] and [1,1,8]. The one with dist 12 is a class mismatch. So still 2 TPs. Recall = 2/5 = 0.4
    assert abs(results['PA@10px'] - 0.4) < 1e-6, "PA@10px should be 0.4"
    assert abs(results['PA@20px'] - 0.4) < 1e-6, "PA@20px should be 0.4"
    print("   Metric calculations are correct.")


if __name__ == "__main__":
    print("============================================")
    print("  CourtPoints Module Implementation Verify  ")
    print("============================================")

    tests = {
        "Model Creation": test_model_creation,
        "Loss Function": test_loss_function,
        "Validation Metrics": test_validation_metrics,
    }

    results = {name: run_test(func, name) for name, func in tests.items()}
    
    print("\n--- Test Summary ---")
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")

    num_passed = sum(results.values())
    num_total = len(results)
    print(f"\nResults: {num_passed}/{num_total} tests passed.")

    if num_passed == num_total:
        print("\n🎉 All new implementations verified successfully!")
    else:
        print("\n⚠️  Some implementations have issues. Please check the errors above.")
