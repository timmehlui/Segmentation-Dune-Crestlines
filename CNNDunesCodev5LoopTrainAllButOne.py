# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:27:40 2024

@author: timmylui

Inspired by examples from segmentation_models binary segmentation example.
Past models weren't working too well, building from ground up.
"""

### Import stuff

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import cv2
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
import segmentation_models as sm
import albumentations as A

### Helper functions for data visualization

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

### Classes for data loading and preprocessing

class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['crestline', 'not_crestline']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
class Dataloder(tf.keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)   
            
class SegmentationDataset:
    def __init__(self, images, masks, transform=None, augmentation=None, 
            preprocessing=None):

        self.images = images
        self.masks = masks
        self.transform = transform
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask, div=1)
            
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
    
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0, rotate_limit=360, shift_limit=0, p=1, border_mode=0),
        
        A.CLAHE(p=1),
        
        #A.RandomBrightnessContrast(contrast_limit = 0.2, p = 0.5),
        
        #A.ColorJitter(brightness = (100, 200), p=0.5),
        
        A.Lambda(mask=round_clip_0_1)
        
        #A.Normalize()

    ]
    return A.Compose(train_transform)

"""
        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                #A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                #A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                #A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        
        A.Lambda(mask=round_clip_0_1)
"""

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

indices_barchanoidal = [2, 5, 11, 12, 1]
scores_list = []

for test_index in indices_barchanoidal:
    print("[INFO] Currently on test_index:" + str(test_index))
    train_list = indices_barchanoidal.copy()
    train_list.remove(test_index)

    indices_train = train_list.copy()

    # Variables
    patch = (256, 256)
    #patch = (128, 128)
    h, w = patch
    c = 1
    
    # Outputs
    images_train_list = []
    masks_train_list = []
    images_test_list = []
    masks_test_list = []

    # Load images
    gs_test = Image.open("/Users/timmylui/OneDrive - Stanford/Documents/PhD/Courses/GEOLSCI220/Data/raster_exports/mars"+str(test_index)+".tif")
    cl_test = Image.open("/Users/timmylui/OneDrive - Stanford/Documents/PhD/Courses/GEOLSCI220/Data/raster_exports/mars"+str(test_index)+"_crestlines.tif")
    
    ## Create test images
    
    # Turn images into numpy arrays
    gs_test_np = np.array(gs_test)
    cl_test_np = np.array(cl_test)
    
    # Apply a mask to turn data into 0 or 1
    cl_mask_test = cl_test_np > 0
    cl_test_np = cl_mask_test.astype(int)
    
    # Turn numpy into tensor
    gs_tn_test = torch.tensor(gs_test_np).reshape(1, 1, gs_test_np.shape[0], gs_test_np.shape[1])
    cl_tn_test = torch.tensor(cl_test_np).reshape(1, 1, gs_test_np.shape[0], gs_test_np.shape[1])
    
    # Create sub images of the size of the patch
    gs_un_test = gs_tn_test.unfold(2,h,128).unfold(3,w,128).transpose(1,3).reshape(-1, c, h, w)
    cl_un_test = cl_tn_test.unfold(2,h,128).unfold(3,w,128).transpose(1,3).reshape(-1, c, h, w)
    
    #gs_un_test = gs_tn_test.unfold(2,h,64).unfold(3,w,64).transpose(1,3).reshape(-1, c, h, w)
    #cl_un_test = cl_tn_test.unfold(2,h,64).unfold(3,w,64).transpose(1,3).reshape(-1, c, h, w)
    
    # Makes sure that images are not incomplete parts of the circle. That there
    # aren't empty pixels
    for i in range(len(gs_un_test)):
        if bool(gs_un_test[i].min() != 0):
            images_test_list.append(np.array(gs_un_test[i][0,:,:]))
            masks_test_list.append(np.array(cl_un_test[i][0,:,:]))
    
    ## Create train images
    
    # Loop over every other image
    for train_index in indices_train:
        gs_train = Image.open("/Users/timmylui/OneDrive - Stanford/Documents/PhD/Courses/GEOLSCI220/Data/raster_exports/mars"+str(train_index)+".tif")
        cl_train = Image.open("/Users/timmylui/OneDrive - Stanford/Documents/PhD/Courses/GEOLSCI220/Data/raster_exports/mars"+str(train_index)+"_crestlines.tif")
        
        # Turn images into numpy arrays
        gs_train_np = np.array(gs_train)
        cl_train_np = np.array(cl_train)
        
        # Apply a mask to turn data into 0 or 1
        cl_mask_train = cl_train_np > 0
        cl_train_np = cl_mask_train.astype(int)
    
        # Turn numpy into tensor
        gs_tn_train = torch.tensor(gs_train_np).reshape(1, 1, gs_train_np.shape[0], gs_train_np.shape[1])
        cl_tn_train = torch.tensor(cl_train_np).reshape(1, 1, gs_train_np.shape[0], gs_train_np.shape[1])
        
        # Create sub images of the size of the patch
        gs_un_train = gs_tn_train.unfold(2,h,128).unfold(3,w,128).transpose(1,3).reshape(-1, c, h, w)
        cl_un_train = cl_tn_train.unfold(2,h,128).unfold(3,w,128).transpose(1,3).reshape(-1, c, h, w)
        #gs_un_train = gs_tn_train.unfold(2,h,64).unfold(3,w,64).transpose(1,3).reshape(-1, c, h, w)
        #cl_un_train = cl_tn_train.unfold(2,h,64).unfold(3,w,64).transpose(1,3).reshape(-1, c, h, w)
        
        # Make sure images don't contain incomplete parts of the circle
        for i in range(len(gs_un_train)):
            if bool(gs_un_train[i].min() != 0):
                images_train_list.append(np.array(gs_un_train[i][0,:,:]))
                masks_train_list.append(np.array(cl_un_train[i][0,:,:]))
                
    # Create images and masks
    images_train_list, images_val_list, masks_train_list, masks_val_list = train_test_split(images_train_list, masks_train_list, test_size=0.2, shuffle = True)
    
    images_train = np.array(images_train_list).astype(np.uint8)
    images_val = np.array(images_val_list).astype(np.float32)
    images_test = np.array(images_test_list).astype(np.float32)
    masks_train = np.array(masks_train_list).astype(np.float32)
    masks_val = np.array(masks_val_list).astype(np.float32)
    masks_test = np.array(masks_test_list).astype(np.float32)
    
    # Turn images from GreyScale to RGB
    images_train = np.repeat(images_train[..., np.newaxis], 3, -1)
    images_val = np.repeat(images_val[..., np.newaxis], 3, -1)
    images_test = np.repeat(images_test[..., np.newaxis], 3, -1)



    # Parameters
    BACKBONE = 'efficientnetb3'
    BATCH_SIZE = 8
    CLASSES = ['crestline']
    LR = 7e-5
    EPOCHS = 100
    
    preprocess_input = sm.get_preprocessing(BACKBONE)
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid'
    
    model = sm.Unet(BACKBONE, classes=1, activation=activation)
    
    # define optomizer
    optim = tf.keras.optimizers.Adam(LR)
    
    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    
    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 
    
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    
    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)
    
    
    
    # Create Datasets
    train_dataset = SegmentationDataset(images_train, masks_train, augmentation = get_training_augmentation())
    val_dataset = SegmentationDataset(images_val, masks_val, augmentation = get_validation_augmentation())
    test_dataset = SegmentationDataset(images_test, masks_test, augmentation = get_validation_augmentation())
    
    # Create Dataloaders
    train_dataloader = Dataloder(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = Dataloder(val_dataset, batch_size=1, shuffle=False)
    
    
    """
    callbacks = [
        keras.callbacks.ModelCheckpoint('./best_model.weights.h5', save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
    ]
    """
    
    # train model
    history = model.fit_generator(
        train_dataloader, 
        steps_per_epoch=len(train_dataloader), 
        epochs=EPOCHS, 
        #callbacks=callbacks, 
        validation_data=val_dataloader, 
        validation_steps=len(val_dataloader),
    )
    
    # Plot training & validation iou_score values
    plt.figure(figsize=(30, 5))
    plt.subplot(121)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    test_dataset = SegmentationDataset(images_test, masks_test)
    
    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
    
    #model.load_weights('best_model.weights.h5') 
    
    scores = model.evaluate_generator(test_dataloader)
    
    print("Loss: {:.5}".format(scores[0]))
    scores_list.append(scores[0])
    for metric, value in zip(metrics, scores[1:]):
        print("mean {}: {:.5}".format(metric.__name__, value))
        scores_list.append(value)
        
    n = 10
    
    test_ids = np.random.choice(np.arange(len(test_dataset)), size=n)
    train_ids = np.random.choice(np.arange(len(train_dataset)), size=n)
    val_ids = np.random.choice(np.arange(len(val_dataset)), size=n)
        
    # Visualization train set
    for i in train_ids: 
        image, true_mask = train_dataset[i]
        image_plot = image[:, :, 0]
        image = np.expand_dims(image, axis=0)
        predicted_mask = model.predict(image).round()
        
        visualize(
            image=image_plot,
            true_mask=true_mask,
            predicted_mask=predicted_mask[0, :, :, 0],
        )
    
    # Visualization val set
    for i in val_ids: 
        image, true_mask = val_dataset[i]
        image_plot = image[:, :, 0]
        image = np.expand_dims(image, axis=0)
        predicted_mask = model.predict(image).round()
        
        visualize(
            image=image_plot,
            true_mask=true_mask,
            predicted_mask=predicted_mask[0, :, :, 0],
        )
        
    # Visualization test set
    for i in test_ids: 
        image, true_mask = test_dataset[i]
        image_plot = image[:, :, 0]
        image = np.expand_dims(image, axis=0)
        predicted_mask = model.predict(image).round()
        
        visualize(
            image=image_plot,
            true_mask=true_mask,
            predicted_mask=predicted_mask[0, :, :, 0],
        )
    
"""
Debugging visualization

n = 5
for i in range(5):
    image_val, mask = val_dataset[i]
    visualize(image = image[:, :, 0], mask = mask)

image_train, _ = train_dataset[0]
image_val, _ = val_dataset[0]
image_test, _ = test_dataset[0]

visualize(image = image_train[:, :, 0])
visualize(image = image_val[:, :, 0])
visualize(image = image_test[:, :, 0])

"""