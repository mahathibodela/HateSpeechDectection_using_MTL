# HateSpeechDetection_using_MTL

This project follows MTL approch by leverages knowledge gained from sentiment analysis and emotion detection in detection hate speech in the given sentence. Out of all the models build, MTL with emotion and polarity outperformed all others.

## RESULTS

| MODEL                           | F1_SCORE |
|---------------------------------|----------|
| STL                             | 66.4%   |
| MTL with emotion                | 71.47%  |
| MTL with polarity               | 70.50%  |
| MTL with emotion and polarity   | 72.27%  |

## Usage
1. You can visit my [application](https://huggingface.co/spaces/Mahathi7/HateSpeech)
2. Input a sentence and click submit

## CHECKPOINTS
You can download the model [checkpoints](https://huggingface.co/spaces/Mahathi7/HateSpeech/blob/main/hateSpeechEmotion.pth)

This project is based on the [reaseach paper](https://ieeexplore.ieee.org/abstract/document/9509436)
