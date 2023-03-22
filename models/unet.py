import segmentation_models_pytorch as smp

def createUnet(inputchannels=1, outputchannels=1):
  model = smp.Unet(
      encoder_name="resnet152",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
      encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
      in_channels=inputchannels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
      activation = 'sigmoid',
      classes=outputchannels,                      # model output channels (number of classes in your dataset)
  )
  model.eval()
  return model

def createUnetPlusPlus(inputchannels=1, outputchannels=1):
  model = smp.UnetPlusPlus('resnet152',
                           encoder_weights="imagenet",
                           in_channels=inputchannels,
                           activation = 'sigmoid',
                           classes=outputchannels)
  return model