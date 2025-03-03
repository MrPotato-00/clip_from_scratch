# clip_from_scratch

This project aims to explore implementation of Clip from scratch with its incredible features of text-to-image and image-to-text retrieval. Make sure cuda pytorch with cuda is already installed in your system (although pytorch with cuda will be installed in this implementation within the virtual environment). A detailed explanation of the implementation will be soon available in a medium article.
```
    class_names =["t-shirt/top",
                        "trousers",
                        "pullover",
                        "dress",
                        "coat",
                        "sandal",
                        "shirt",
                        "sneaker",
                        "bag",
                        "ankle boot"]
```
These are the query for the image-to-text retrieval.
![Screenshot from 2025-03-03 23-59-24](https://github.com/user-attachments/assets/cd2554b7-85b6-46da-a97c-adc889e2fe63)

![Screenshot from 2025-03-03 23-54-25](https://github.com/user-attachments/assets/f7c5dceb-ff85-4aeb-822e-6d59db02c17a)

Below is the query of text-to-image retrieval:
![Screenshot from 2025-03-03 23-55-04](https://github.com/user-attachments/assets/3b821075-e666-45d8-ab2f-09106ab81785)
![Screenshot from 2025-03-03 23-56-18](https://github.com/user-attachments/assets/7f031a97-15ea-40a1-98ae-247e829678f3)
![Screenshot from 2025-03-03 23-56-50](https://github.com/user-attachments/assets/48e0e5fd-a7d5-4aee-b121-b48d70a1c2e5)
![Screenshot from 2025-03-03 23-57-34](https://github.com/user-attachments/assets/ad573247-48ae-4499-8546-1dd1cae8591f)
![Screenshot from 2025-03-03 23-57-47](https://github.com/user-attachments/assets/4440ac87-f895-43a7-8965-eac531715823)
![Screenshot from 2025-03-03 23-58-17](https://github.com/user-attachments/assets/0bf7d08b-b4fc-479a-a2de-3c4ad0ff495d)



Steps for running testing the model. 
1. Clone this repository.
2. Create a virtual environment.
3. Install the dependencies. (pip install -r requirements.txt)
4. Run the training script. (python3 main.py)
5. Run the retrieval script while passing argument in command line.

A sample example of image-to-text retrieval:




The image-to-text implementation was inspired from the implementation of clip from scratch using this medium article: https://medium.com/correll-lab/building-clip-from-scratch-68f6e42d35f4

If you have any have any suggestion for improvement create a pull request.
