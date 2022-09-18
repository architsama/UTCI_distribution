import torchvision.models
import torch
import torchsummary
import model_M

model = model_M.model_6

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# model.to(device)

torchsummary.summary(model(),(1,30,30
                              ),device='cpu')



#  21:  11,343,409
#   1:  48,627,900


