from models.ewc import EWC

# After phase-1 training
ewc = EWC(
    model=model,
    dataset=train_dataset,
    device=device,
    fisher_sample_size=500
)

# During phase-2 training
loss = task_loss + ewc_lambda * ewc.penalty(model)

