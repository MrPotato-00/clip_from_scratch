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
These are the query for the text-to-image retrieval.

Steps for running testing the model. 
1. Clone this repository.
2. Create a virtual environment.
3. Install the dependencies. (pip install -r requirements.txt)
4. Run the training script. (python3 main.py)
5. Run the retrieval script while passing argument in command line.


The image-to-text implementation was inspired from the implementation of clip from scratch using this medium article: https://medium.com/correll-lab/building-clip-from-scratch-68f6e42d35f4

If you have any have any suggestion for improvement create a pull request.
