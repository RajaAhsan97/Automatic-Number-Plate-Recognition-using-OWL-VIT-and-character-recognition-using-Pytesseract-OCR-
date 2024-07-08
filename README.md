This repository contains two python code files for recognizing text on the license plate using Pytesseract OCR engine. Prior to text recognition, the first step is to localize the license 
plate (LP) in the given input image. File "anpr.py" contains the conventional image processing techniques to localize the license plate, while the file "anpr_owl-vit.py" contains a pre-trained 
model OWL-ViT by hugging face [ref: https://huggingface.co/docs/transformers/en/model_doc/owlvit] for license plate localization. Localization of license plate using conventional techniques 
results in detection of non-license plate regions. Whereas by using OWL-ViT only the regions containing the license plate is recognized. After localizing the LP, the region is fed to the 
Optical Character Recognition (OCR) engine by PyTesseract. Refer to the below images to visualize the results, where the first image shows the license plate localization using conentional image processing techniques and the second image contains the localized license plate region using OWL-ViT.

*    IMAGE 1
  
![result](https://github.com/RajaAhsan97/Automatic-Number-Plate-Recognition-using-OWL-VIT-and-character-recognition-using-Pytesseract-OCR-/assets/155144523/ff409242-6536-4b7c-8921-65a64000df37)

![Cars0](https://github.com/RajaAhsan97/Automatic-Number-Plate-Recognition-using-OWL-VIT-and-character-recognition-using-Pytesseract-OCR-/assets/155144523/7e6cf9a0-ce81-401b-9b79-eaae39011820)

_____________________________________________________________________________________________________________________________________________________________________________________
*    IMAGE 2
  
![result](https://github.com/RajaAhsan97/Automatic-Number-Plate-Recognition-using-OWL-VIT-and-character-recognition-using-Pytesseract-OCR-/assets/155144523/d175f239-c5b7-46f7-8324-c1ca29a1190f)

![Cars1](https://github.com/RajaAhsan97/Automatic-Number-Plate-Recognition-using-OWL-VIT-and-character-recognition-using-Pytesseract-OCR-/assets/155144523/85f3f29a-84e9-43ee-88ba-175765439887)


From the results of image 1, it could be visualized that while OCRing the localized license plate region the character 'L' and 'C' is recognized as 'E' and 'O' by tesseract OCR engine. 
This is due to the localized region needs to be further preprocessed for filtering before processing it with tesseract OCR engine. 
