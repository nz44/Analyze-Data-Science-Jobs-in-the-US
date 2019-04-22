# Final-Project-Group3
2019 Spring DATS 6103_10 

If you want to save time, the data is downloaded, extracted and included in the code folder.
You can clone them to your workspace and start running 1_Main_Script.py from PyCharm.

If you want to download the data from its original source on Kaggle, please follow the instructions below:


Please install Kaggle API according to 
https://github.com/Kaggle/kaggle-api/blob/master/README.md

run in command on windows
pip install kaggle
run on mac/linux
pip install --user kaggle

on windows, navigate to C:\Users\<Windows-username>\
on cloud, in your home folder. 
Create a kaggle file on your windows location
mkdir .kaggle\


Place the Kaggle.json file in the Code folder inside the Kaggle file described above.

Run below in command line (Unix):
chmod 600 ~/.kaggle/kaggle.json


navigate to project folder
Download the dataset by running in command line:
kaggle datasets download -d sl6149/data-scientist-job-market-in-the-us


On Unix, run in command:
unzip data-scientist-job-market-in-the-us.zip
On Windows, unzip by right clikcing "extract all"


The unzipped data is "alldata.csv", it should appear in the project folder. 

Open 1_Main_Script.py from PyCharm, run

You should see all the plots and printout on the python console or terminal.

