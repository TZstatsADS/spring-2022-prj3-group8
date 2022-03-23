### Doc folder

The doc directory contains the report or presentation files. 

main.ipynb contains all the combined work from each team member. The Other main files with name inthe parentheses show work teams members built in personally.

3rdparty folders contains semi-supervised learning code base we used to generate clean labels. Several changes have been made to the repo in order to run the algorithm on our custom dataset. 
All experiments are ran via configuration file, i. e. you can run training on custom dataset with command:
python flexmatch.py --c ./configs/custom/exp01_baseline.yaml
