# -*- coding: utf-8 -*-
from util.freq_han import freq_han

cho = "ㄱㄲㄳㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
jung = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
SPECIALS = "♡♥★☆"  # len = 2

hanguel_len = len(freq_han)
dic_han = {freq_han[k]: k for k in range(hanguel_len)}
def decompose_as_one_hot(in_char, warning=True):
    # Return : [1,2350] : 음절, [2351,2478] : 아스키, [2479, 2529]: 단모음, 단자음 [2530, 2533] : SPECIAL, [2534] : Unknown
    if chr(in_char) in freq_han:  # 음절 인코딩에 포함될때.
        result = dic_han[chr(in_char)]
    elif in_char < 128:  # 아스키
        result = hanguel_len + in_char
    elif ord('ㄱ') <= in_char <= ord('ㅣ'):  # 단모음, 단자음
        result = hanguel_len + 128 + (in_char - 12593)
    elif chr(in_char) in SPECIALS:
        result = hanguel_len + 128 + 51 + SPECIALS.index(chr(in_char))
    else:
        if warning:
            print('Unhandled character:', chr(in_char), in_char)
        # unknown character
        result = hanguel_len + 128 + 51 + len(SPECIALS)
    return [result + 1]

def decompose_str_as_one_hot_eum(string, warning=True):
    tmp_list = []
    for x in string:
        da = decompose_as_one_hot(ord(x), warning=warning)
        tmp_list.extend(da)
    return tmp_list
