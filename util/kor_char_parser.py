# -*- coding: utf-8 -*-
cho = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"  # len = 19
jung = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"  # len = 21
jong = "ㄱ/ㄲ/ㄱㅅ/ㄴ/ㄴㅈ/ㄴㅎ/ㄷ/ㄹ/ㄹㄱ/ㄹㅁ/ㄹㅂ/ㄹㅅ/ㄹㅌ/ㄹㅍ/ㄹㅎ/ㅁ/ㅂ/ㅂㅅ/ㅅ/ㅆ/ㅇ/ㅈ/ㅊ/ㅋ/ㅌ/ㅍ/ㅎ".split('/') # len = 27
han_len = len(cho) + len(jung) + len(jong)
special = "0123456789,;.!?:'""/\|_@#$%^&*~`+-=<>()[]{}ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ" # len = 80
special_dic = {}
for i, k in enumerate(special):
    special_dic[k] = i

def decompose_str_as_one_hot(string, warning=False):
    tmp_list = []
    for x in string:
        da = decompose_as_one_hot(ord(x), warning)
        if da[0] == 9999:
            continue
        tmp_list.extend(da) # == tmp_list += da
    return tmp_list


def decompose_as_one_hot(in_char, warning=False):
    one_hot = []
    if ord('가') <= in_char <= ord('힣'):  # 가:44032 , 힣: 55203
        x = in_char - 44032  # in_char - ord('가')
        y = x // 28
        z = x % 28
        x = y // 21
        y = y % 21
        one_hot.append(x)
        one_hot.append(len(cho) + y)
        if z > 0:
            one_hot.append(len(cho) + len(jung) + (z - 1)) # if there is jong, then is z > 0. So z starts from 1 index.
        return one_hot
    else:
        if chr(in_char) in special:
            return[han_len + special_dic[chr(in_char)]]
        else:
            if warning:
                print('Unhandled character:', chr(in_char), in_char)
            # unknown character
            return [9999]