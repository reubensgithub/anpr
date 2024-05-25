- The Django folder is within the ANPR subfolder
- The train dataset used to train the character recognition model is called new_tr_dataset_ref
- The validation dataset used to validate the character recognition model is called new_tr_valid_dataset_ref
- The test dataset used to test the character recognition model is called new_tr_test_dataset_ref
- best_tr_model2 contains the weights of the custom model. The same folder is situated within the ANPR subfolder.
- best_tr_pretrained_model contains the weights of the VGG16 transfer learning model.
- reworked_tr_model.ipynb is the notebook I developed the character recognition model on Google Colab on.
- The venv for this project was deleted due to its size, and the pre-requisites of the project can be found within requirements.txt in the ANPR subfolder. This was created using pip freeze > requirements.txt
To run the server. open terminal or command prompt, cd into the ANPR subfolder and run the command:
'python manage.py runserver'

