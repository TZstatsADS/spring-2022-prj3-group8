## Project 3 
### Weakly supervised learning: label noise and correction


Team members: Sibo Geng, Xiran Lin, Joel Mugyenyi, Rhea Sablani, Jiuru Wang

Summary: In this project, we developed two models, one for training the labels of the images with the given labels, and another for training the labels with semi-supervised learning. We achieved 96% accuracy when training with semi-supervised learning and 68% accuracy when training with the predicted clean labels.

[Contribution Statement] 

Joel and Rhea built CNN models for Model I and Model II using both a simple CNN model and a pretrained model (RESNet). Rhea worked on the construction and optimization of the CNN model that had an improved prediction accuracy compared to the baseline logistic model.

Sibo, Xiran, and Jiuru worked together as a team on label cleansing. They discussed and tried several models on **torchssl**, and by comparison, found that **flexmatch** is the best among the models we tried. 

All team members contributed to the GitHub repository and prepared the presentation. All team members approve our work presented in our GitHub repository including this contribution statement.
