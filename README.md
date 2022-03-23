# Project: Weakly supervised learning-- label noise and correction


### [Full Project Description](doc/project3_desc.md)

Term: Spring 2022

+ Team 8
+ Team members
	+ Sibo Geng (sg4010@columbia.edu)
	+ Xiran Lin (xl3000@columbia.edu)
	+ Joel Mugyenyi (jm5352@columbia.edu)
	+ Rhea Sablani (rss2229@columbia.edu)
	+ Jiuru Wang (jw4150@columbia.edu)

+ Project summary: In this project, we use *logistic regression* as our baseline model. We developed a model, which contains 8 layer, and trained this model with noisy labels. We got approximately 22% accuracy as a result. Then, we used FlexMatch, a semi-supervised learning algorithm, to generate the predicted clean label from the images and their noisy labels, which yields about 96% validation accuracy of the labels. We put the *"predicted cleaned labels"* into our model and did the same training, after all we got about 68% accuracy, which is way better than the one with noisy labels. 
	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) All team members contributed equally in all stages of this project. All team members approve our work presented in this GitHub repository including this contributions statement. 
