import re

def clean_text(text: str) -> str:
    # Remove UNK_BYTE tags
    text = re.sub(r'\[UNK_BYTE_0x[0-9a-fA-F]+\]', '', text)

    # Remove literal backslashes
    text = text.replace("\\", "")

    # Remove standalone "uf"
    text = re.sub(r'\buf\b', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

print(clean_text("[UNK_BYTE_0xe996a2関][UNK_BYTE_0xe382acガ][UNK_BYTE_0x7f][UNK_BYTE_0xe3839eマッチ][UNK_BYTE_0xe38383マッチ][UNK_BYTE_0xe38381マッチ][UNK_BYTE_0xe5a4a7大きな][UNK_BYTE_0xe3818d大きな][UNK_BYTE_0xe381aa大きな][UNK_BYTE_0xe682aa悪い][UNK_BYTE_0xe38184悪い][UNK_BYTE_0xe383b3ン][UNK_BYTE_0xe382acガチャ][UNK_BYTE_0xe38381ガチャ][UNK_BYTE_0xe383a3ガチャ][UNK_BYTE_0xe383aaリサイクル][UNK_BYTE_0xe382b5リサイクル][UNK_BYTE_0xe382a4リサイクル][UNK_BYTE_0xe382afリサイクル][UNK_BYTE_0xe383abリサイクル][UNK_BYTE_0xe69dbf板][UNK_BYTE_0xe99ba3難][UNK_BYTE_0xe381b8へん][UNK_BYTE_0xe38293へん][UNK_BYTE_0xe38188えば][UNK_BYTE_0xe381b0えば][UNK_BYTE_0xe29480─]uf[UNK_BYTE_0xe4b88d不完全][UNK_BYTE_0xe5ae8c不完全][UNK_BYTE_0xe585a8不完全]"""))
