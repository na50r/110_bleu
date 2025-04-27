from os.path import exists, join
from scripts.util import get_env_variables, store_sents, MyLogger, LANG_ISO
from io import BytesIO
from abc import ABC, abstractmethod
from typing import Any


def get_deepl_client() -> Any:
    from deepl import DeepLClient
    api_key = get_env_variables('DEEPL_API_KEY')
    client = DeepLClient(auth_key=api_key)
    return client


def get_openai_client() -> Any:
    from openai import OpenAI
    api_key = get_env_variables('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)
    return client


class TranslationClient(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def translate_document(self, text: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        '''
        Args:
            text: A list of sentences/strings
            src_lang: ISO code for source language
            tgt_lang: ISO code for target language

        Returns:
            A list of translated sentences, will ideally contain the same number of strings as input
        '''
        pass


class DeeplClient(TranslationClient):
    TGT_LANGS = {
        # Account for special target language codes
        'en': 'EN-GB',
        'pt': 'PT-PT'
    }

    def __init__(self,  logger: MyLogger | None = None):
        self.client = get_deepl_client()
        self.logger = logger
        self.model = 'deepl_document'

    # Input is a list of sentences
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
                src_text=in_text,
                translator=self.model
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
    def __init__(self, model: str = 'gpt-4.1', logger: MyLogger | None = None):
        self.client = get_openai_client()
        self.logger = logger
        self.model = model

    def sys_prompt(self, src_lang: str, tgt_lang: str):
        # System prompt based on https://github.com/jb41/translate-book/blob/main/main.py
        p = f"You are a {LANG_ISO[src_lang]}-to-{LANG_ISO[tgt_lang]} translator."
        return p

    def user_prompt(self, src_lang: str, tgt_lang: str, text: str):
        p1 = f"Translate the following {LANG_ISO[src_lang]} sentences into {LANG_ISO[tgt_lang]}."
        p2 = f"Please make sure to keep the same formatting, do not add more newlines."
        p3 = f"Here is the text:"
        return '\n'.join([p1, p2, p3, text])

    def chat_complete(self, sys_prompt: str, user_prompt: str):
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': user_prompt},
            ]
        )

        # Log real GPT tokens
        if self.logger and resp.usage is not None:
            self.logger.finish(
                tgt_text=resp.choices[0].message.content,
                in_model_tokens=resp.usage.prompt_tokens,
                out_model_tokens=resp.usage.completion_tokens,
                finish_reason=resp.choices[0].finish_reason
            )
        return resp.choices[0].message.content

    def translate_document(self, text: list[str], src_lang: str, tgt_lang: str):
        # Input is a list of sentences
        # Input for GPT4.1 are system and user prompts
        in_text = '\n'.join(text)
        # TikToken count will be smaller than real since system & user prompt excluded
        if self.logger:
            self.logger.start(
                src_text=in_text,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                translator=self.model)

        sys_prompt = self.sys_prompt(src_lang, tgt_lang)
        user_prompt = self.user_prompt(src_lang, tgt_lang, in_text)
        trans_text = self.chat_complete(sys_prompt, user_prompt)
        return trans_text.splitlines()


def translate_document(text: list[str], src_lang: str, tgt_lang: str, client: TranslationClient, mt_folder: str | None = None) -> list[str] | None:
    '''
    Main translation function
    This function returns translations but also stores them in the specified folder
    If run again by accident, will not call API if translation is detected in the specified folder

    Args:
        text: A list of sentences/strings
        src_lang: ISO code for source language
        tgt_lang: ISO code for target language
        client: A translator client that has a translate_document method specified by TranslationClient abstract class
        mt_folder: Path to folder where translations should be stored

    Returns:
        A list of translated sentences, will ideally contain the same number of strings as input
    '''
    if mt_folder is None:
        return client.translate_document(
            text=text,
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )
    
    out_file = join(mt_folder, f'{src_lang}-{tgt_lang}.txt')
    if not exists(out_file):
        out_text = client.translate_document(
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
        print(
            f'Document for pair {src_lang}-{tgt_lang} has been translated already.')
        return None
