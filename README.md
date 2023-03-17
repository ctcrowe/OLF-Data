# OLF-Data
A simple Transformer Network used to predict the Occupant Load Factor in a given room based upon the name of that room. The project only has 3 files.
<br><br>
<b>Model.py</b> contains all of the code for the entire project including loading testing and training the model.
<br>
<b>OLFNetworkData.txt</b> is the training dataset for the model. This will be the most commonly updated file in the project
<br>
<b>OLFNetwork.pt</b> is the actual model created from the code. This can get reimplemented into revit events to automatically load rooms and run qc on the occupancy tables for a revit model.
