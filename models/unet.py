import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

def createUnet(inputchannels=1, outputchannels=1, is_train=False):
  model = smp.Unet(
      encoder_name="resnet152",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
      encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
      in_channels=inputchannels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
      activation='sigmoid' if not is_train else None,
      classes=outputchannels,                      # model output channels (number of classes in your dataset)
  )
  if not is_train:
    model.eval()
  return model

def createUnetPlusPlus(inputchannels=1, outputchannels=1):
  model = smp.UnetPlusPlus('resnet152',
                           encoder_weights="imagenet",
                           in_channels=inputchannels,
                           activation = 'sigmoid',
                           classes=outputchannels)
  return model


preprocessing_fn = get_preprocessing_fn(encoder_name="resnet152", pretrained="imagenet")