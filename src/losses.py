import torch
import torch.nn as nn

class MultiTaskCrossEntropyLoss(nn.Module):
    """
    Calculates the total loss for a multi-task classification problem.
    It is the sum of the cross-entropy losses for each task.
    """
    def __init__(self):
        super(MultiTaskCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """
        Args:
            outputs (list of Tensors): A list of output tensors from the model. 
                                       Each tensor corresponds to a task and has 
                                       shape (batch_size, num_classes_for_task).
            targets (Tensor): A tensor of target labels with shape (batch_size, num_tasks).
        
        Returns:
            Tensor: The total loss, which is the sum of the losses for each task.
        """
        losses = []
        num_tasks = len(outputs)
        
        for i in range(num_tasks):
            task_output = outputs[i]
            task_target = targets[:, i]
            loss = self.criterion(task_output, task_target)
            losses.append(loss)
            
        total_loss = torch.stack(losses).sum()
        return total_loss 