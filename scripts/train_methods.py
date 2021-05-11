import torch
import torch.nn as nn
import time
import numpy as np
import wandb

def train(config,
          model,
          optimizer,
          loss_function,
          train_dataloader,
          device,
          history,
          verbose=0):
    # training mode for layers like dropout, batchnorm etc in training mode
    model.train()
    train_loss = []
    predictions_dict = {}
    start_time = time.time()

 
    # Iterate through the data set given
    for batch_idx, (data, labels) in enumerate(train_dataloader):
        end_time = time.time()
        dataloader_time = end_time - start_time

        # Load the data and load to gpu
        start_time = time.time()
        data, labels = data.to(device), labels.to(device)
        end_time = time.time()
        gpu_time = end_time - start_time

        optimizer.zero_grad()
        start_time = time.time() 
        output = model(data)
        end_time = time.time()
        f_propagation_time = end_time - start_time

        # loss function, and compute the loss
        loss = loss_function(output, labels)

        # gradients
        start_time = time.time()
        loss.backward()
        end_time = time.time()
        b_propagation_time = end_time - start_time

        # gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()

        # append only loss  
        train_loss.append(loss)
        start_time = time.time()
        if verbose == 1:
            print("\nBatch: ", batch_idx)
            print("Dataloader time: ", dataloader_time)
            print("GPU loading time: ", gpu_time)
            print("Forward propagation time: ",f_propagation_time)
            print("Backward propagation time: ",b_propagation_time)

    # append metrics to history dict
    history["loss"].append(sum(train_loss) / len(train_loss))

    print("Train Loss {%f}"%(history["loss"][-1]))

    

def validate(config,
             model,
             loss_function,
             validation_dataloader,
             device,
             history,
             run=None):
    # Switch model to evaluation mode
    model.eval()
    loss = 0
    correct = 0
    val_loss = []
    predictions_dict = {}


    # load dataloader
    # validation_dataloader = DataLoader(validation_dataset, batch_size=config.test_batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=False)

    example_images = []
    with torch.no_grad():
        for data, labels in validation_dataloader:
            # Load the data and labels from the test dataset
            data, labels = data.to(device), labels.to(device)
            output = model(data)

            # loss function, and compute the loss
            loss = loss_function(output, labels)

            # append loss
            val_loss.append(loss)


     # append metrics to history dict
    history["val_loss"].append(sum(val_loss) / len(val_loss))

    print("Validation Loss {%f}"% (history["val_loss"][-1]) )

    # log at wandb these metrics
    if run is not None:
        # log segmentation mask from last mini batch
        mask_images = []
        for image, label, mask_data in zip(data, labels, output):
            image, label, mask_data = np.squeeze(image.cpu().numpy()), np.squeeze(label.cpu().numpy()), np.squeeze(mask_data.cpu().numpy())
            # conver predictions between 0 and 1
            mask_data = np.where(mask_data > mask_data.mean(), 1, 0)
            label = np.where(label > label.mean(), 1, 0)

            class_labels = {1: "lane"}

            mask_img = wandb.Image(image, masks={
            "predictions": {"mask_data": mask_data, "class_labels": class_labels},
            "groud_truth": {"mask_data": label, "class_labels": class_labels}
            })
            mask_images.append(mask_img)
            
        run.log(
            { 
            "loss": history["loss"][-1],
            "val_loss": history["val_loss"][-1],
            "model_predictions": mask_images
             }
            )
        # run.log({"model_predictions": [wandb.Image(mask_img, caption="Label")] })
    
def save_chechpoint(checkpoint_path, model, optimizer, epoch, history):
    torch.save({
            'initial_epoch': epoch,
            'model': model,
            'optimizer_state_dict': optimizer,
            'history': history
            }, checkpoint_path)
        
def train_model(model=None,
                optimizer=None,
                loss=None,
                config=None,
                train_dataloader=None,
                validation_dataloader=None,
                device=None,
                run=None,
                history=None,
                checkpoint_path=None,
                initial_epoch=1,
                log_interval=5,
                verbose=0):
    

    # if not resume the history
    if history is None:
        history = {"loss": [], "val_loss": [],
        "title": "SimpleLaneNet model\nl_rate ["+str(config.learning_rate)+"] batch_size ["+str(config.batch_size)+"] reg ["+str(config.weight_decay)+"] drop ["+str(config.dropout)+"] epochs ["+str(config.epochs)+"]"}
    
    # set the learning rate of the optimizer
    for parameters in optimizer.param_groups:
        parameters['lr'] = config.learning_rate

    start = time.time()
    for epoch in range(config.epochs):
        print(epoch)
        train(config, model, optimizer, loss, train_dataloader, device, history, verbose)
        validate(config, model, loss, validation_dataloader, device, history, run)

        # save a checkpoint after log interval
        if (checkpoint_path is not None) and (epoch%log_interval==0):
            save_chechpoint(checkpoint_path, model, optimizer, epoch, history)
            # log artifact to wandb
            wandb_log_artifact(run=run, artifact_name=checkpoint_path, file=checkpoint_path, type_="model")
    
    # log finally at the end
    save_chechpoint(checkpoint_path, model, optimizer, epoch, history)
    wandb_log_artifact(run=run, artifact_name=checkpoint_path, file=checkpoint_path, type_="model")
    
    end = time.time()
    print("Total time: "+str(end-start)+" time per epoch: "+str((end-start)/config.epochs))

    # history["model"] = model
    torch.cuda.empty_cache()

    return history