---
date:
  created: 2025-05-21
---

# Cross Entropy

## Introduction

This blog will cover a very simple exercise, coding Cross entropy loss from scratch such that the loss value for a random `(input,target)` pair from our implementation is the same as the loss from PyTorch’s cross Entropy implementation

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b984b9cf-d72e-4aae-9b3e-0153dd84e0f6/4a4a2330-4125-4266-9b25-c790f1e5f332/image.png)

### Pseudo code

- We will have a input logits of shape (Batch, number of classes) . For our sample we have a batch size of 3 and 5 classes.
- The targets will be integers of shape (batch, 1) . They will be integers between [0, num_classes -1]
- logits can be negative as well and in any scale ! not necessary between 0 and 1
- Step 1: We have to apply softmax for each row ( for each sample in a batch)
- Step 2: Choose the likelihood corresponding to target class and apply
  `-log(probability_of_correct_class)` on it
- Step 3: take the mean of all log probabilities in a batch

### Avoid Integer overflow !

Note that to calculate softmax we have to take the exponent of the logits. For logits as big as 1000, the computer doesn’t have enough bits to store this large number! To avoid this we do a neat trick

In simple Words, **subtracting a constant from the logits , won’t change their softmax probability**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b984b9cf-d72e-4aae-9b3e-0153dd84e0f6/84c87b51-d934-4498-889f-85d9ca556810/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b984b9cf-d72e-4aae-9b3e-0153dd84e0f6/ec95f334-0dbf-47d5-8ffa-e597b71196d4/image.png)

As you can see, the e^c terms cancel out in both numerator and denominator, proving that subtracting any constant (in our case, the maximum value) from all logits gives us the same softmax probabilities, but in a numerically stable way.

The key step is that e^{x-c} = e^x/e^c, which lets the common factor e^c cancel out in the final division.

### Code

```python
import torch
import numpy as np

def cross_entropy_loss(predictions, targets):
    """
    Custom implementation of cross entropy loss

    Args:
        predictions: Raw model output logits of shape (batch_size, num_classes)
        targets: Ground truth labels of shape (batch_size,)

    Returns:
        loss: Mean cross entropy loss across the batch
    """
    # Get batch size
    batch_size = predictions.shape[0]

    # Apply softmax to get probabilities
    exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    softmax_preds = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

    # Get predicted probability for the correct class
    correct_class_probs = softmax_preds[range(batch_size), targets]

    # Calculate negative log likelihood
    loss = -np.log(correct_class_probs + 1e-7)  # Add small epsilon for numerical stability

    # Return mean loss
    return np.mean(loss)

# Test the implementation
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    batch_size = 3
    num_classes = 4

    # Create random logits and targets
    logits = np.random.randn(batch_size, num_classes)
    targets = np.random.randint(0, num_classes, size=batch_size)

    # Calculate loss using our implementation
    custom_loss = cross_entropy_loss(logits, targets)

    # Calculate loss using PyTorch
    torch_logits = torch.FloatTensor(logits)
    torch_targets = torch.LongTensor(targets)
    torch_loss = torch.nn.CrossEntropyLoss()(torch_logits, torch_targets)

    print("Input logits:")
    print(logits)
    print("\nTarget labels:", targets)
    print("\nCustom implementation loss:", custom_loss)
    print("PyTorch implementation loss:", torch_loss.item())
    print("\nDifference:", abs(custom_loss - torch_loss.item()))
```
