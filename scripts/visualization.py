import numpy as np
import torch
import cv2
import time
import matplotlib.pyplot as plt
from matplotlib import gridspec
from google.colab.patches import cv2_imshow



def autoencoder_visualization(histories, train_data, model, device, num_of_test_images=4,fontsize=15):

    # set plot surface
    num_of_test_images+=1
    num_of_histories = len(histories)
    # if num_of_histories > 1:
    num_of_histories += 1
    num_of_train_data = len(train_data)
    x_dim = train_data[0][0].shape[2]
    y_dim = train_data[0][0].shape[1]
    c_dim = train_data[0][0].shape[0]
    fig = plt.figure(figsize=(20,11*num_of_histories))
    # fig = plt.figure(figsize=(10,4*num_of_histories))
    fig.suptitle("Visualization of Loss with random True and their Predicted images for Every Experiment", fontsize=fontsize+5)

    gs  = gridspec.GridSpec(num_of_test_images*num_of_histories, 3, width_ratios=[0.66, 0.165, 0.165], height_ratios=np.ones(num_of_test_images*num_of_histories))
    ax0 = [plt.subplot(gs[i*num_of_test_images:i*num_of_test_images+num_of_test_images-1, 0]) for i in range(num_of_histories-1)]
    ax_true = [plt.subplot(gs[i,1]) for i in range(num_of_test_images*(num_of_histories-1))]
    ax_pred = [plt.subplot(gs[i,2]) for i in range(num_of_test_images*(num_of_histories-1))]


    # plot loss and validation loss
    for history in range(num_of_histories-1):
        ax0[history].plot(histories[history]['loss'], label ='loss')
        ax0[history].plot(histories[history]['val_loss'], label='val loss')
        ax0[history].set_xlabel('Epoch', fontsize=fontsize)
        ax0[history].set_ylabel('Loss', fontsize=fontsize)
        ax0[history].legend(fontsize=fontsize)
        ax0[history].grid(True)
        for val in (histories[history]['loss'], histories[history]['val_loss'])[1:]:
            ax0[history].annotate('%0.5f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'))
        # get model info
        # model_info = print_model_info_autoencoder(histories[history]['model_info'])
        # ax0[history].set_title("Experiment: "+str(history+1)+"\n"+model_info+"\nModel loss", fontsize=fontsize)

        # show true random images
        random_test_indexes = np.random.randint(0, num_of_train_data-1, num_of_test_images)
        for ax in range(num_of_test_images-1):
            ax_true[ax+history*num_of_test_images].imshow(train_data[random_test_indexes[ax]][0].reshape(x_dim,y_dim),  cmap='gray')
            ax_true[ax+history*num_of_test_images].set_title("True image", fontsize=fontsize)
            ax_true[ax+history*num_of_test_images].axis("off")
        ax_true[num_of_test_images-1+history*num_of_test_images].axis("off")

        # get the predictions
        prediction = [model.get_output(train_data[index][0].to(device)) for index in random_test_indexes]
        for ax in range(num_of_test_images-1):
            ax_pred[ax+history*num_of_test_images].imshow(prediction[ax],  cmap='gray')
            ax_pred[ax+history*num_of_test_images].set_title("Predicted image", fontsize=fontsize)
            ax_pred[ax+history*num_of_test_images].axis("off")
        ax_pred[num_of_test_images-1+history*num_of_test_images].axis("off")

    ### plot all together ###
    if num_of_histories > 2:
        ax = plt.subplot(gs[(num_of_histories-1)*num_of_test_images:, 0])

        for history in range(num_of_histories-1):
            ax.plot(histories[history]['val_loss'], label="experiment "+str(history+1))
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Epoch', fontsize=fontsize)
        ax.set_ylabel('Loss', fontsize=fontsize)
        ax.set_title("Models' losses", fontsize=fontsize)

        # scatter experimets
        ax = plt.subplot(gs[(num_of_histories-1)*num_of_test_images:, 1:])

        for history in range(num_of_histories-1):
            ax.scatter(history, histories[history]['loss'][-1], label="exp "+str(history+1)+" loss", marker="o")
            ax.scatter(history, histories[history]['val_loss'][-1], label="exp "+str(history+1)+" val loss", marker="X")
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Experiments', fontsize=fontsize)
        ax.set_ylabel('Loss', fontsize=fontsize)
        ax.set_xticks(range(num_of_histories-1))
        ax.set_xticklabels(["exp "+str(x+1) for x in range(num_of_histories-1)])
        ax.set_title("Models' losses per hyperparameters", fontsize=fontsize)


    # plt.close()
    # _ = fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show(block=True)
    # return fig


def video_creator(file_name,
                  dataset_images, 
                  start_frame, 
                  frames, 
                  labels=False, 
                  fps=24, 
                  overlay_opacity=0.2):    

    frame = dataset_images[start_frame][0]
    # frame = np.array(frame, dtype=np.float32)
    height, width, channels = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(file_name+".avi", fourcc, fps, (width,height))

    for i in range(start_frame, start_frame+frames):
        # write video
        frame = dataset_images[i][0]
        # if display overlay lables
        if labels:
            frame = cv2.addWeighted(frame, 1, dataset_images[i][1], 1, 0)
        out.write(frame) 

    # save file
    out.release()

# Overlay image with predicted labels from a detector
def video_detector_creator(file_name,
                  detector,
                  dataset_images, 
                  start_frame, 
                  frames, 
                  labels=False, 
                  fps=24, 
                  overlay_opacity=0.2):    

    frame = dataset_images[start_frame][0]
    # frame = np.array(frame, dtype=np.float32)
    height, width, channels = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(file_name+".avi", fourcc, fps, (width,height))

    for i in range(start_frame, start_frame+frames):
        # write video
        frame = dataset_images[i][0]

        # overlay detected lines
        detected_lines = detector.get_output(frame)
        frame = cv2.addWeighted(frame, 1, detected_lines, 1, 0)

        # if display overlay labels
        if labels:
            frame = cv2.addWeighted(frame, 1, dataset_images[i][1], 1, 0)

        out.write(frame) 

    # save file
    out.release()

# handle different dataset between torch and cv2
def torch_video_detector_creator(file_name,
                  detector,
                  dataset_images, 
                  pytorch_dataset_images, 
                  start_frame, 
                  frames, 
                  labels=False, 
                  fps=24, 
                  overlay_opacity=0.2):    

    frame = dataset_images[start_frame][0]
    height, width, channels = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(file_name+".avi", fourcc, fps, (width,height))

    for i in range(start_frame, start_frame+frames):
        # write video
        frame = dataset_images[i][0]

        # overlay detected lines
        detected_lines = detector.get_output(pytorch_dataset_images[i][0])
        frame = cv2.addWeighted(frame, 1, detected_lines, 1, 0)

        # if display overlay labels
        if labels:
            frame = cv2.addWeighted(frame, 1, dataset_images[i][1], 1, 0)

        out.write(frame) 

    # save file
    out.release()


def image2grid(images, texts, grid):
    # code from https://gist.github.com/pgorczak/95230f53d3f140e4939c
    import itertools

    w = grid[0]
    h = grid[1]
    n = w*h
    img_h, img_w, img_c = images[0].shape
    m_x = 8
    m_y = 40

    font = cv2.FONT_HERSHEY_SIMPLEX
  
    # image offset
    x_offset = -60
    fontScale = 0.4        
    fontScale = 0.6
    thickness = 2
    
    imgmatrix = np.zeros((img_h * h + m_y * (h - 1) + m_y, img_w * w + m_x * (w - 1), img_c), np.uint8)
    imgmatrix.fill(255)    

    positions = itertools.product(range(h), range(w))
    for (y_i, x_i), img, text in zip(positions, images, texts):
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        # add text
        imgmatrix[y:y+img_h, x:x+img_w, :] = img
        cv2.putText(imgmatrix, text, (int(img_w/2)+x+x_offset, y+img_h+int(m_y/2)), font, fontScale, (255,0,0), thickness, cv2.LINE_AA)

    return imgmatrix


# Overlay images on a grid with predicted labels from a detector
def grid_video_detector_creator(file_name,
                  detectors,
                  texts,
                  dataset_images, 
                  start_frame, 
                  frames,
                  grid=(2,2), 
                  labels=False, 
                  fps=24, 
                  overlay_opacity=0.2):
    
    images = [dataset_images[start_frame][0] for image in range(len(detectors))]
    grid_frame = image2grid(images, texts, grid)
    height, width, channels = grid_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(file_name+".avi", fourcc, fps, (width,height))

    for i in range(start_frame, start_frame+frames):
        # write video
        images = [dataset_images[i][0] for image in range(len(detectors))]
        grid_frame = image2grid(images, texts, grid)

        # overlay detected lines
        detected_images = [detector.get_output(dataset_images[i][0]) for detector in detectors]
        detected_grid_frame = image2grid(detected_images, texts, grid)

        frame = cv2.addWeighted(grid_frame, 1, detected_grid_frame, 1, 0)

        # if display overlay labels
        if labels:
            labels_images = [dataset_images[i][1] for image in range(len(detectors))]
            labels_grid_frame = image2grid(labels_images, texts, grid)
            frame = cv2.addWeighted(frame, 1, labels_grid_frame, 1, 0)

        out.write(frame) 

    # save file
    out.release()

# Overlay images on a grid with predicted labels from a detector
def grid_image_detector_creator(file_name,
                  detectors,
                  texts,
                  dataset_images, 
                  start_frame, 
                  grid=(2,2), 
                  labels=False, 
                  fps=24, 
                  overlay_opacity=0.2):
    
    images = [dataset_images[start_frame][0] for image in range(len(detectors))]
    grid_frame = image2grid(images, texts, grid)
    height, width, channels = grid_frame.shape


    # write image
    images = [dataset_images[start_frame][0] for image in range(len(detectors))]
    grid_frame = image2grid(images, texts, grid)

    # overlay detected lines
    detected_images = [detector.get_output(dataset_images[start_frame][0]) for detector in detectors]
    detected_grid_frame = image2grid(detected_images, texts, grid)

    frame = cv2.addWeighted(grid_frame, 1, detected_grid_frame, 1, 0)

    # if display overlay labels
    if labels:
        labels_images = [dataset_images[start_frame][1] for image in range(len(detectors))]
        labels_grid_frame = image2grid(labels_images, texts, grid)
        frame = cv2.addWeighted(frame, 1, labels_grid_frame, 1, 0)

    cv2.imwrite(file_name, frame) 


def torch_imshow(iamge):
    cv2_imshow(iamge.permute(1, 2, 0).detach().numpy())
    
def torch_model_to_cv2(image, range=255):
    # check if is not 3channel
    if image.size()[0] == 1:
        image = image.repeat(3,1,1)
    image_size = image.size()
    image = image.view(image.size(0), -1)
    image_min = image.min()
    image_max = image.max()
    # print(image_min, image_max)
    image += -image_min
    image *= range/(image_max - image_min)
    image = image.view(image_size)
    # permute channel to cv2
    image = image.view(image_size)
    image = image.permute(1,2,0).detach().numpy().astype(np.uint8)
    return image

def torch_imshow_model_output(image):
    cv2_imshow( torch_model_to_cv2(image))