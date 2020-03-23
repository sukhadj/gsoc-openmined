import torch 

from simpleNN import SimpleNN

from core import scheme, models
from utils import fix_precision, float_precision 

model = SimpleNN(10)

print("Loading model ...")
path = "/home/sukhad/Workspace/GithHub/reading-in-the-dark/mnist/objects/ml_models/simple_char.pt" 
model.load_state_dict(torch.load(path))
model.eval()
print("Done")


proj_prec = 7
diag_prec = 5
data_prec = 3

model.proj1.weight = fix_precision(model.proj1.weight, proj_prec, 100)
model.proj1.bias = fix_precision(model.proj1.bias, proj_prec, 100)

model.diag1.weight = fix_precision(model.diag1.weight, diag_prec, 100)

proj_param = torch.cat((model.proj1.bias.reshape((1,-1))/2**data_prec, model.proj1.weight.t())).long().tolist()
diag_param = model.diag1.weight.t().long().tolist()

print("Loading proj and forms ...")
proj = models.Projection(proj_param)
forms = models.DiagonalQuadraticForms(diag_param)
print("Done")

model_path = "/home/sukhad/Workspace/GithHub/reading-in-the-dark/mnist/objects/ml_models"
ml = models.MLModel(proj, forms)
ml.toFile(model_path, "simple_nn_quad")