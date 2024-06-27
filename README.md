# TCL
### Introduction
> To further improve the 
recommendation performance, this paper proposes an effective 
Web API recommendation approach via exploring textual and 
structural semantics with Trustworthy contrastive learning, 
named TCL. 

### Environment Requirment
> This code has been tested running undeer Python 3.9.0
> The Required packages are as follows:
> - torch == 2.0.0+cu118
> - numpy == 1.24.1
> - seaborn == 0.13.2
> - transformers ==4.37.2
> - wheel ==0.41.2

### Example to run TCL
 - Command`python train_TCL.py`
 - Train log:
>    开始训练    
存在训练数据，正在加载    
存在测试数据，正在加载    
 10%|█         | 500/5000 [00:09<07:29, 10.02it/s]    
> 
NOTE : the duration of training and testing depends on the running environment.   
Train environment is on CPU AMD R5 5600x GPU RTX4060ti.    
### File Introduction
1. model.py
> This file contains the code of TCL.
2. sanfm.py
> This file contains the code of sanfm.
3. utils.py
> This file contains the founction used in the item.
4. train_TCL.py
> This file is the model training file.
5. dataset.py
> This file contains the dataset loading code.
6. you need to download uncased-bert to the root of this item.    
7. data is available in [here](https://pan.baidu.com/s/16fvcMfva8mew662O4XvIAQ?pwd=8mam)


