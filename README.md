# GPT Failure Branch
* This branch was left to 'prove' that there was indeed a repeated issue with the language pair Dutch-to-Danish which GPT4.1 repeatedly refused to translate fully.
* The prompt user prompt used can be found in the code as well but was:
```
Translate the following English sentences into German.
Please make sure to keep the same formatting, do not add more newlines.
Here is the text:
```
* The main branch thus started to use the updated prompt:
```
Translate the following English sentences into German.
Please make sure to keep the same formatting, do not add more newlines.
You are not allowed to omit anything.
Here is the text:
```
* It is worth noting that the original prompt DID work in some cases, such as running `translate_document` from the `GPTClient` directly in another Jupyter Notebook.
* However, it kept failing in the notebook where we wanted it to work, so we opted to just add the prompt. 
