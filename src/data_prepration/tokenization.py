def word_tokenizer(texts: list, tokenizer_obj) -> list:
    """

    :param texts: list of sents ex: ["first sent", "second sent"]
    :param tokenizer_obj:
    :return: list of tokenized words ex: [["first", "sent"], ["second", "sent"]]
    """
    return [tokenizer_obj(text) for text in texts]


def sent_tokenizer(texts: list, tokenizer_obj):
    """

    :param texts: list of docs ex: ["first sent. second sent.", "first sent. second sent."]
    :param tokenizer_obj:
    :return: list of tokenized sents ex: [["first sent.", "second sent."],
                                        ["first sent.", "second sent."]]
    """
    return [tokenizer_obj(text) for text in texts]
