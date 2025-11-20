def test_model(model, test_loader, device):
    test_acc = calculate_accuracy(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    return test_acc
