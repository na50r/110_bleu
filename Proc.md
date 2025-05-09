# Procedure Documentation
Here we document the different procedures based on `procedure.py` that we run in this project.

## Procedure 1
It acts as an initial test of the codebase and was used to translate 20 pairs of sentences from datasets OPUS100, EuroPalr and Flores+ using translators GPT4.1 and DeepL. It worked without any issues but we did observe a substantial difference in translation time between GPT4.1 and Deepl. We also observed that GPT4.1 has it's own retry mechanism. The following 6 tasks were run in this procedure:
```
python proc1.py run -m gpt-4.1-2025-04-14 -d opus-100
python proc1.py run -m gpt-4.1-2025-04-14 -d europarl
python proc1.py run -m gpt-4.1-2025-04-14 -d flores_plus
python proc1.py run -m deepl_document -d opus-100
python proc1.py run -m deepl_document -d europarl
python proc1.py run -m deepl_document -d flores_plus
```
Each 20 language pairs per task, 400 sentences. The logs and Jupyter Notebook were uplouded to GitHub but we decided for the following the next procedures to not do this anymore, as logs can grow rather large.

## Procedure 2
The plan was to first translate the remaining 90 pairs for sentences from EuroParl and Flores+ using GPT4.1. Since we knew it would take too much time to translate all 90 at once, we decided to split it into half, thus 4 tasks, where we initially translate the first and second for EuroParl and then the same for Flores+. This showed us an issue with the design of the procedure, as it is not flexible to splitting the selected pairs in this manner. The pairs can be selected by defining the script but execution is bounded on dataset and translator. However, at this point we did not want to refactor our code nor redo the translation, so we just distinguished the different halves by appending a `-1` or `-2` to the dataset identifier. This resulted in 4 tasks that could be run but only one was run in the end:
```
python proc2.py run -m gpt-4.1-2025-04-14 -d europarl-1
```
We observed that it took GPT4.1 roughly 244 minutes or more than 4 hours to translate all 45 pairs. This happened primarily due to two instances where it got stuck with retrying the API call due 504 Gateway Timeout errors. We also observed a potential bug if `europarl-2` was run. The `task.json` file is hardcoded and generated in the folder where the translations are stored. In this case, both `europarl-1` and `europarl-2` store translations in the same folder, meaning the `task.json` would be overwritten. Hence we fix the bug and run the remaining tasks in Procedure 3 instead. 

## Procedure 3
