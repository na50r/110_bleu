from os.path import exists, join
from scripts.util import get_env_variables, store_sents, load_sents, LANG_ISO
from scripts.logger import TranslationLogger
from io import BytesIO
from abc import ABC, abstractmethod
from deepl import DeepLClient
from openai import OpenAI
from string import Template, ascii_letters, ascii_lowercase, ascii_uppercase
from typing import Any
import logging
from scripts.data_management import DataManager
from scripts.constants import R1, R2, R3, E


class TranslationClient(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def translate_document(self, text: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        pass

    def translate_and_store_document(self, text: list[str], src_lang: str, tgt_lang: str, mt_folder: str | None = None) -> list[str]:
        '''
        This method returns translations but also stores them in the specified folder
        If run again by accident, will not call API if the translation is detected in the specified folder

        Args:
            text: A list of strings (sentences)
            src_lang: ISO code of source language
            tgt_lang: ISO code of target language
            client: A client that has a translate_document method specified by TranslationClient abstract class
            mt_folder: Path to a folder where translations should be stored

        Returns:
            A list of translated strings (sentences), will ideally contain the same number of strings as input
        '''
        if mt_folder is None:
            return self.translate_document(
                text=text,
                src_lang=src_lang,
                tgt_lang=tgt_lang
            )

        out_file = join(mt_folder, f'{src_lang}-{tgt_lang}.txt')
        if not exists(out_file):
            out_text = self.translate_document(
                text=text,
                src_lang=src_lang,
                tgt_lang=tgt_lang
            )
            store_sents(
                sents=out_text,
                folder_path=mt_folder,
                src_lang=src_lang,
                tgt_lang=tgt_lang
            )
            return out_text
        else:
            logging.info(
                f'[📋]: Pair {src_lang}-{tgt_lang} has been translated already.')
            return load_sents(mt_folder, src_lang, tgt_lang)


class DeeplClient(TranslationClient):
    TGT_LANGS = {
        # Account for special target language codes
        'en': 'EN-GB',
        'pt': 'PT-PT'
    }

    def __init__(self,  logger: TranslationLogger | None = None):
        api_key = get_env_variables('DEEPL_API_KEY')
        self.client = DeepLClient(auth_key=api_key)
        self.logger = logger
        self.model = 'deepl_document'

    # Input is a list of strings (sentences)
    # Input for translate_document requires a 'document', in this case, 'bytes'
    def translate_document(self, text: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        out_buffer = BytesIO()
        in_text = '\n'.join(text)
        in_bytes = in_text.encode('utf-8')
        in_filename = 'infile.txt'

        if self.logger:
            self.logger.start(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_text=in_text
            )

        resp = self.client.translate_document(
            input_document=in_bytes,
            output_document=out_buffer,
            source_lang=src_lang.upper(),
            target_lang=DeeplClient.TGT_LANGS.get(tgt_lang, tgt_lang.upper()),
            filename=in_filename
        )

        out = out_buffer.getvalue()
        out_text = out.decode('utf-8')

        if self.logger:
            self.logger.finish(
                tgt_text=out_text,
                status=resp.status.value,
                error_msg=resp.error_message
            )

        out_sents = out_text.splitlines()
        return out_sents


class GPTClient(TranslationClient):
    # Using Templates for prompt for development purposes
    # Based on: https://stackoverflow.com/questions/11630106/advanced-string-formatting-vs-template-strings
    SYS_TEMPL = 'You are a professional translation system.'

    USR_TEMPL = Template(
        "Translate the following $src_lang sentences into $tgt_lang.\n"
        "Please make sure to keep the same formatting, do not add more newlines.\n"
        "You are not allowed to omit anything.\n"
        "Here is the text:\n"
        "$text")

    HYPER_PARAMS = {'temperature': 0, 'seed': 42}

    def __init__(self, model: str = 'gpt-4.1-2025-04-14',
                 logger: TranslationLogger | None = None,
                 sys_templ: Template | str | None = SYS_TEMPL,
                 usr_templ: Template | str | None = USR_TEMPL,
                 hyper_params: dict[str, Any] = HYPER_PARAMS
                 ):
        api_key = get_env_variables('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        self.logger = logger
        self.model = model
        self.sys_templ = sys_templ
        self.usr_templ = usr_templ
        self.hyper_params = hyper_params

    def sys_prompt(self, src_lang: str, tgt_lang: str) -> str | None:
        if self.sys_templ is None:
            return None

        if isinstance(self.sys_templ, str):
            return self.sys_templ

        p = self.sys_templ.substitute(
            src_lang=LANG_ISO[src_lang],
            tgt_lang=LANG_ISO[tgt_lang]
        )
        return p

    def user_prompt(self, src_lang: str, tgt_lang: str, text: str) -> str | None:
        if self.usr_templ is None:
            return None

        if isinstance(self.usr_templ, str):
            return self.usr_templ

        p = self.usr_templ.substitute(
            src_lang=LANG_ISO[src_lang],
            tgt_lang=LANG_ISO[tgt_lang],
            text=text
        )
        return p

    def print_config(self, src_lang='$src_lang', tgt_lang='$tgt_lang', text='$text'):
        print(f'Hyperparameters: {self.hyper_params}')
        print('\nSystem Prompt:')
        if isinstance(self.sys_templ, str) or self.sys_templ is None:
            print(self.sys_templ)
        else:
            print(self.sys_templ.substitute(
                src_lang=src_lang, tgt_lang=tgt_lang))
        print('\nUser Prompt:')
        if isinstance(self.usr_templ, str) or self.usr_templ is None:
            print(self.usr_templ)
        else:
            print(self.usr_templ.substitute(
                tgt_lang=tgt_lang, src_lang=src_lang, text=text))

    def chat_complete(self, sys_prompt: str, user_prompt: str) -> str:
        msgs = []
        if sys_prompt is not None:
            msgs.append({'role': 'system', 'content': sys_prompt})

        if user_prompt is not None:
            msgs.append({'role': 'user', 'content': user_prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            **self.hyper_params
        )
        logging.debug(f'HTTP Response: {resp.model_dump()}')
        # Logs real GPT tokens & GPT specific resp messages
        if self.logger and resp.usage is not None:
            self.logger.finish(
                tgt_text=resp.choices[0].message.content,
                in_model_tokens=resp.usage.prompt_tokens,
                out_model_tokens=resp.usage.completion_tokens,
                finish_reason=resp.choices[0].finish_reason,
                system_fingerprint=resp.system_fingerprint)
        return resp.choices[0].message.content

    def translate_document(self, text: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        # Input is a list of strings (sentences)
        # Input for GPT4.1 are system and user prompts
        in_text = '\n'.join(text)
        # Token count computed by Logger will be smaller as system & user prompt are excluded
        if self.logger:
            self.logger.start(
                src_text=in_text,
                src_lang=src_lang,
                tgt_lang=tgt_lang)

        sys_prompt = self.sys_prompt(src_lang, tgt_lang)
        user_prompt = self.user_prompt(src_lang, tgt_lang, in_text)
        trans_text = self.chat_complete(sys_prompt, user_prompt)
        return trans_text.splitlines()


class MockClient(TranslationClient):
    '''
    MockClient is a client that can be used to test the translation task.
    It can be configured to raise errors, return incorrect translations, or return correct translations.
    It can also be configured to simulate a scenario, where a list of integers represents the scenario to be simulated (scenario code in constants.py).
    If DataManager is provided, it will be used to generate correct (perfect) translations, should be used if language detection is tested as well.
    If no DataManager is provided, it will use a Caesar Cipher to generate translations, should be used if language detection is not tested.
    
        Args:
            logger: A TranslationLogger that logs translation specific information
            model: A string that can be used to use to identify the same client as other clients
            planned_rejects: A list of tuples of source and target language ISO codes that will be rejected
            planned_errors: A list of tuples of source and target language ISO codes that will raise an error
            scenario: A list of integers that represents the scenario to be simulated 
        '''

    def __init__(self, dm: DataManager = None, logger=None, model='mock', planned_rejects=[], planned_errors=[], scenario=[]):
        opt1 = len(scenario) >= 0 and len(planned_errors+planned_rejects) == 0
        opt2 = len(scenario) == 0 and len(planned_errors+planned_rejects) >= 0,
        assert opt1 or opt2, 'Please provide either a scenario alone or planned rejects and errors'

        opt3 = (R2 in scenario or R3 in scenario) and dm is not None
        opt4 = (R2 not in scenario and R3 not in scenario)
        assert opt3 or opt4, 'Please provide a DataManager if you want to simulate R2 (wrong language) or R3 (no language detected)'

        self.dm = dm
        self.logger = logger
        self.model = model
        self.planned_rejects = planned_rejects
        self.planned_errors = planned_errors
        self.scenario = scenario
        self.current = 0

    def encrypt(self, text: str, key: int = 13, direction: int = 1, pair: tuple[str, str] = None):
        # Code from: https://stackoverflow.com/a/34734063
        if direction == -1:
            key = 26 - key

        trans = str.maketrans(
            ascii_letters, ascii_lowercase[key:] + ascii_lowercase[:key] + ascii_uppercase[key:] + ascii_uppercase[:key])

        out_text = text.translate(trans)

        if (len(self.scenario) > 0 and self.scenario[self.current] == R1) or pair in self.planned_rejects:
            if pair in self.planned_rejects:
                self.planned_rejects.remove(pair)
            tmp = out_text.splitlines()
            tmp = tmp[:round(len(tmp)/2)]
            out_text = '\n'.join(tmp)
        return out_text

    def get_sent_pairs(self, text, pair):
        num_of_sents = len(text.splitlines())
        if (len(self.scenario) > 0 and self.scenario[self.current] == R1) or pair in self.planned_rejects:
            if pair in self.planned_rejects:
                self.planned_rejects.remove(pair)
            num_of_sents = round(num_of_sents/2)

        if len(self.scenario) > 0 and self.scenario[self.current] == R2:
            pair = (pair[1], pair[0])

        if len(self.scenario) > 0 and self.scenario[self.current] == R3:
            out_text = '\n'.join(['.']*400)  # edge case that triggers error
            return out_text

        _, out_text = self.dm.get_sentence_pairs(
            *pair, num_of_sents=num_of_sents)
        return '\n'.join(out_text)

    def translate(self,  src_lang: str, tgt_lang: str, text: str) -> str:
        '''
        Flexible method that either runs Caesar Cipher or uses DataManager to get translations
        '''
        pair = (src_lang, tgt_lang)
        if len(self.scenario) > 0 and self.scenario[self.current] == E:
            self.current += 1  # an error log will be created in the except statement, so we increment the current translation here
            raise (Exception(f'MockError'))

        if pair in self.planned_errors:
            self.planned_errors.remove(pair)
            raise (Exception(f'MockError'))

        if self.dm is not None:
            out_text = self.get_sent_pairs(text, pair)
        else:
            out_text = self.encrypt(text, key=13, direction=-1, pair=pair)
        return out_text

    def translate_document(self, text: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        in_text = '\n'.join(text)
        if self.logger:
            self.logger.start(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_text=in_text)
        out_text = self.translate(src_lang, tgt_lang, in_text)
        if self.logger:
            self.logger.finish(tgt_text=out_text)
            self.current += 1
        return out_text.splitlines()
