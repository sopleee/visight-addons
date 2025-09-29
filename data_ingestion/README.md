## Data Ingestion (first pass)
Run reformat_raw.py in Modal. It saves data formatted according to current a2.tex into s3. 

1) There is an error in the Roboflow SDK- filenames in our dataset are sometimes too long, and it causes an error. So, I currently downloaded the dataset into my local device (manually pressing continue every time a filename was too long). A sample of said dataset is included to get a sense of the dataset structure. 
2) reformat_raw.py accomplishes two things: create and save brand_catalogue and image_catalogue to s3.