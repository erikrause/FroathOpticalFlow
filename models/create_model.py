import torch

from models.attunet import create_attunet, AttU_Net
from models.transunet import createTransUnet
from models.dstransunet import  createDSTransUnet
from models.unet import createUnet, createUnetPlusPlus

"""
     Returns Unet model.
     Parameters:
     Returns:
      :torch.Model - model for training
         """
def createModel(model_name):
  if model_name=='unet':
    model = createUnet()
  elif model_name=='unetplusplus':
    model = createUnetPlusPlus()
  elif model_name=='attunet':
    model = create_attunet()
  elif model_name=='transunet':
    model = createTransUnet()
  elif model_name=='dstransunet':
    model = createDSTransUnet()
  else:
    raise ValueError('no such model')
  return model


def evalModel(weights_path):
  model = torch.load(weights_path, map_location=torch.device('cpu'))
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()
  return model