# PneumCNN

A Convolutional Neural Network approach for diagnosis of Pneumonia from chest X-ray reports.

## Getting Started
1. Clone this repository
2. To set the virtual environment and install dependencies run
   ```
    virtualenv -p python3 venv  # to be created only once
    source venv/bin/activate
    pip install -r requirements.txt
   ```
3. To run the streamlit app, run
   ```
   streamlit run streamlit.py
   ```

## Project Layout

`cnn.ipynb`: is the main notebook that has the network model, The model is saved after training into a `JSON` format, 
and the weights are saved in an `h5` file.

`load_model.py`: has a function `test` which loads the model and the weights.
Then it predicts whether the input image (in the form of an `numpy.ndarray`) is pneumonic or not

`streamlit.py`: is a minimal GUI made using [Streamlit](https://streamlit.io/). Users can upload an image in the app. 
The app invokes the `test` function from the `load_model.py` file and passes the image in the form of a `numpy.ndarray`
