# Automatic Number Plate Recognition (ANPR) Project

## Project Structure

- **Django Application**: The Django folder is located within the `ANPR` subfolder.
- **Datasets**:
  - Training dataset: `new_tr_dataset_ref`
  - Validation dataset: `new_tr_valid_dataset_ref`
  - Test dataset: `new_tr_test_dataset_ref`
- **Model Weights**:
  - Custom model weights: `best_tr_model2` (located within the `ANPR` subfolder)
  - VGG16 transfer learning model weights: `best_tr_pretrained_model`
- **Development Notebook**: The notebook used to develop the character recognition model is `reworked_tr_model.ipynb` (developed on Google Colab).
- **Dependencies**: The virtual environment for this project was deleted due to its size. The project prerequisites can be found in `requirements.txt` within the `ANPR` subfolder. This file was created using `pip freeze > requirements.txt`.

## Running the Server

To run the server, open a terminal or command prompt, navigate to the `ANPR` subfolder, and execute the following command:

```bash
python manage.py runserver
```

## Examples
![Example 1](https://github.com/reubensgithub/anpr/blob/main/detection_segmentation.png)
