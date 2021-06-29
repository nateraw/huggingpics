# 🤗🖼️HuggingPics

Fine-tune Vision Transformers for **anything** using images found on the web.

## Usage

Click on the link below to try it out:

<a href="https://colab.research.google.com/github/nateraw/huggingpics/blob/main/HuggingPics.ipynb" target="_parent\"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## How does it work?

### 1. You define your search terms, we download ~150 images for each

![pick search terms](images/pick_search_terms.png)

### 2. We download ~150 images for each and use them to fine-tune a ViT

![image search results](images/image_search_results.png)

### 3. You push your model to HuggingFace's Hub to share your results with the world

![push to hub](images/push_to_hub.png)


## Examples

💡 If you need some inspiration, take a look at the examples below:


|            | [nateraw/rare-puppers](https://huggingface.co/nateraw/rare-puppers) | [nateraw/pasta-pizza-ravioli](https://huggingface.co/nateraw/pasta-pizza-ravioli) | [nateraw/baseball-stadium-foods](https://huggingface.co/nateraw/baseball-stadium-foods) | [nateraw/denver-nyc-paris](https://huggingface.co/nateraw/denver-nyc-paris) |
| ---------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **term_1** | samoyed                                                             | pizza                                                                             | cotton candy                                                                            | denver                                                                      |
| **term_2** | shiba inu                                                           | pasta                                                                             | hamburger                                                                               | new york city                                                               |
| **term_3** | corgi                                                               | ravioli                                                                           | hot dog                                                                                 | paris                                                                       |
| **term_4** |                                                                     |                                                                                   | nachos                                                                                  |                                                                             |
| **term_5** |                                                                     |                                                                                   | popcorn                                                                                 |                                                                             |

You can see a full list of model repos created using this tool by [clicking here](https://huggingface.co/models?filter=huggingpics)