# CelebA Dataset Analysis & Twin Finder 

This project consists of two parts: an Exploratory Data Analysis (EDA) of the CelebA dataset and a standalone Desktop Application that uses Machine Learning to find celebrity lookalikes from user photos.


## How to Run the Desktop App:
Due to GitHub's file size limits (100MB), the pre-computed FaceNet embeddings for the CelebA dataset (`all_embeddings.npy` - ~400MB) are not included in this repository.

**To run the application locally, please follow these steps:**
1. Download the required database files (`all_embeddings.npy`, `image_names.txt`, `list_attr_celeba.txt`) from our cloud storage: **https://disk.360.yandex.ru/d/a2RaQwLzhn-gFw**

   All 3 files total about 1.5 GB. The mandatory files to download are all_embeddings.npy and list_attr_celeba.txt — these allow the program to find your lookalikes. The img_align_celeba folder is also required if you want to see the actual images of your twins (though it does take up a lot of space).
3. Place all downloaded files in the same root directory as the `app.py` script.
4. Install the required dependencies in your terminal: 
```bash
   pip install flet torch facenet-pytorch opencv-python numpy pandas Pillow
```
4. Run with
```bash
   python app.py
```
