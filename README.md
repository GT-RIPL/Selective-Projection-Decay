# Selective-Projection-Decay

 This repo implements the AdamSPD optimizer in the paper [``Rethinking Weight Decay for Robust Fine-Tuning of Foundation Models``]().

## Use AdamSPD in Your Project
- AdamSPD is the AdamW variant with built-in Selective Projection Decay for **fine-tuning**. It can be easily intergrated into you project for robust fine-tuning of a pre-trained model. Copy the `adamSPD.py` file into your optimizer folder. Here is an example how you would incoroprate the `AdamSPD` optimizer into your project. 

```python
from adamSPD import AdamSPD
optimizer_params = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
} # Initalize optimizer parameters
params_to_opt = [x[1] for x in model.named_parameters() if x[1].requires_grad]
params_anchor = copy.deepcopy(params_to_opt) # Cache pre-trained model weights 
param_group = [{'params':params_to_opt,
                'pre': params_anchor}]
optimizer = AdamSPD(param_group,**optimizer_params)
```
- Working with Parameter-Efficient-Fine-Tuning (PEFT) methods such as [LORA](https://arxiv.org/abs/2106.09685), AdamSPD does not require storing the pre-trained weights. 

```python
from adamSPD import AdamSPD
optimizer_params = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
} # Initalize optimizer parameters
params_to_opt = [x[1] for x in model.named_parameters() if x[1].requires_grad]
param_group = [{'params':params_to_opt,
                'pre': None}]
optimizer = AdamSPD(param_group,**optimizer_params)
```
 