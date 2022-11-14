# Deeplearning-OCR
Using deep learning to read handwritten words 
CONTENTS:
PACKAGES REQUIRED
FILES FOR PROJECT
INSTRUCTIONS FOR PROGRAM SETUP


PACKAGES REQUIRED:
absl-py==1.2.0
altgraph==0.17.3
astunparse==1.6.3
cachetools==5.2.0
certifi==2022.9.14
charset-normalizer==2.1.1
contourpy==1.0.5
cycler==0.11.0
docopt==0.6.2
flatbuffers==2.0.7
fonttools==4.37.4
future==0.18.2
gast==0.4.0
google-auth==2.11.0
google-auth-oauthlib==0.4.6
google-pasta==0.2.0
grpcio==1.49.0
h5py==3.7.0
idna==3.4
keras==2.10.0
Keras-Preprocessing==1.1.2
kiwisolver==1.4.4
libclang==14.0.6
Markdown==3.4.1
MarkupSafe==2.1.1
matplotlib==3.6.0
numpy==1.23.3
oauthlib==3.2.1
opt-einsum==3.3.0
packaging==21.3
pefile==2022.5.30
Pillow==9.2.0
pip==22.2.2
pipreqs==0.4.11
protobuf==3.19.5
pyasn1==0.4.8
pyasn1-modules==0.2.8
pyinstaller==5.5
pyinstaller-hooks-contrib==2022.10
pyparsing==3.0.9
python-dateutil==2.8.2
pywin32-ctypes==0.2.0
requests==2.28.1
requests-oauthlib==1.3.1
rsa==4.9
setuptools==63.2.0
six==1.16.0
tensorboard==2.10.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow==2.10.0
tensorflow-estimator==2.10.0
tensorflow-io-gcs-filesystem==0.27.0
termcolor==2.0.1
typing_extensions==4.3.0
urllib3==1.26.12
Werkzeug==2.2.2
wheel==0.37.1
wrapt==1.14.1
yarg==0.1.9

REQUIRED FILES:

OCR_GUI{
  OCR_GUI.py
  handwriting_recognition.py
  requirements.txt
  data{
  inputs.txt
  words.txt
    inputs{image chosen by user should go in this folder}
    words{Words from the IAMWORDS database will be in here}*
  }
  training_1{
    checkpoint  
  }
  *sidenote: To get the words, you will need to unzip the I_AM_WORDS database folder, and extract words.tgz, then extract words.tar

PROGRAM SETUP:
Our project was made to be able to install all of the packages without needing to manually install all of them to avoid extra work on the user. This is why we have the requirements.txt. After ensuring that the files are all set up properly, you will want to download the packages. You will need to be able to use pip install, and you will use the command "pip install requirements.txt". 
Once the required packages are installed, you will then go into the code. While we made tthe program as user-friendly with a GUI, there was one step that we were unable to get. This part is essential, and why the file setup is so important. In the handwriting_recognition.py you will go down to base_path, or line 26 in github, and change the directory to your own directory where you have the files all saved. Using the layout, you will provide the directory to your data folder. 
Once done with that, the setup is done, nothing needs to be done in the OCR_GUI.py, besides running the project from that. 



